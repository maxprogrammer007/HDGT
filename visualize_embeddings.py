"""
visualize_embeddings.py

HDGT Priority 5 — Embedding Visualisation (UMAP + t-SNE + PCA)

Samples Qwen2.5-VL node embeddings from a subset of graphs, reduces
dimensionality, and produces publication-quality scatter plots coloured by:
  - node role  (paragraph, table, header, caption, ...)
  - page index (within document)
  - document   (which doc the node belongs to)

Outputs are saved to: experiments/embedding_visualizations/

Usage:
    python visualize_embeddings.py                # UMAP (recommended)
    python visualize_embeddings.py --method tsne  # t-SNE
    python visualize_embeddings.py --method pca   # PCA (fast, no install needed)
    python visualize_embeddings.py --method all   # All three
    python visualize_embeddings.py --n_nodes 8000 # Sample size
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("visualize_embeddings")


# ---- colour palette per role ------------------------------------------------
ROLE_COLORS = {
    "paragraph":       "#4C72B0",
    "table":           "#DD8452",
    "header":          "#55A868",
    "figure_caption":  "#C44E52",
    "list_item":       "#8172B2",
    "footnote":        "#937860",
    "title":           "#DA8BC3",
    "page_header":     "#8C8C8C",
    "other":           "#CCCCCC",
}

DOC_PALETTE = plt.cm.tab20.colors  # up to 20 documents


def parse_args():
    p = argparse.ArgumentParser(
        description="HDGT Embedding Visualisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--method", choices=["umap", "tsne", "pca", "all"],
                   default="umap")
    p.add_argument("--n_nodes", type=int, default=6000,
                   help="Total text nodes to sample across all graphs.")
    p.add_argument("--n_graphs", type=int, default=60,
                   help="Number of graphs to sample from.")
    p.add_argument("--graphs_dir", default="experiments/mpdocvqa")
    p.add_argument("--out_dir",    default="experiments/embedding_visualizations")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_node_sample(graphs_dir: Path, n_graphs: int, n_nodes: int,
                     seed: int = 42):
    """
    Randomly sample `n_nodes` text nodes from `n_graphs` graphs.
    Returns (embeddings [N,2048], roles [N], page_ids [N], doc_ids [N]).
    """
    random.seed(seed)
    graph_files = sorted(graphs_dir.glob("*_graph.pt"))
    if not graph_files:
        raise FileNotFoundError(f"No graph files found in {graphs_dir}")

    sampled_files = random.sample(graph_files, min(n_graphs, len(graph_files)))
    logger.info(f"Sampling from {len(sampled_files)} graphs...")

    all_embs, all_roles, all_pages, all_docs = [], [], [], []

    for gpath in sampled_files:
        try:
            g = torch.load(gpath, weights_only=False)
        except Exception as e:
            logger.warning(f"Failed to load {gpath.name}: {e}")
            continue

        if "text" not in g.node_types:
            continue
        txt = g["text"]
        if not hasattr(txt, "qwen_x") or txt.qwen_x is None:
            continue

        mapping = txt._mapping
        embs   = txt.qwen_x.float()   # [N, 2048]
        roles  = mapping.get("roles",  ["other"] * embs.shape[0])
        pages  = mapping.get("pages",  [0]       * embs.shape[0])
        doc_id = gpath.stem.split("_p")[0]

        all_embs.append(embs)
        all_roles.extend(roles)
        all_pages.extend(pages)
        all_docs.extend([doc_id] * embs.shape[0])

    if not all_embs:
        raise RuntimeError("No embeddings found. Run feature extraction first.")

    embs_cat = torch.cat(all_embs, dim=0).numpy()   # [Total, 2048]

    # Subsample
    total = embs_cat.shape[0]
    if total > n_nodes:
        idx = random.sample(range(total), n_nodes)
        embs_cat  = embs_cat[idx]
        all_roles = [all_roles[i] for i in idx]
        all_pages = [all_pages[i] for i in idx]
        all_docs  = [all_docs[i]  for i in idx]

    logger.info(f"  Final sample: {embs_cat.shape[0]:,} nodes from "
                f"{len(set(all_docs))} documents")
    return embs_cat, all_roles, all_pages, all_docs


def reduce_pca(embs: np.ndarray, n_components: int = 2) -> np.ndarray:
    from sklearn.decomposition import PCA
    logger.info("Running PCA...")
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(embs)


def reduce_tsne(embs: np.ndarray) -> np.ndarray:
    from sklearn.manifold import TSNE
    # First reduce to 50D with PCA for speed
    from sklearn.decomposition import PCA
    logger.info("Running PCA(50) → t-SNE(2)...")
    pre = PCA(n_components=min(50, embs.shape[1]), random_state=42)
    embs50 = pre.fit_transform(embs)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000,
                random_state=42, verbose=1)
    return tsne.fit_transform(embs50)


def reduce_umap(embs: np.ndarray) -> np.ndarray:
    try:
        import umap
    except ImportError:
        logger.error("umap-learn not installed. Run: pip install umap-learn")
        raise
    logger.info("Running UMAP...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric="cosine", random_state=42, verbose=True)
    return reducer.fit_transform(embs)


def plot_by_role(coords: np.ndarray, roles: list, method: str, out_dir: Path):
    """Publication-grade UMAP scatter plot coloured by node role."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")

    # Map roles into distinct semantic categories
    cleaned_roles = []
    for r in roles:
        r_str = str(r).lower()
        if "table" in r_str or "cell" in r_str:
            cleaned_roles.append("table")
        elif "head" in r_str or "title" in r_str:
            cleaned_roles.append("header")
        elif "capt" in r_str or "fig" in r_str:
            cleaned_roles.append("figure_caption")
        elif "list" in r_str:
            cleaned_roles.append("list_item")
        elif "foot" in r_str or "note" in r_str:
            cleaned_roles.append("footnote")
        else:
            cleaned_roles.append("paragraph")

    unique_roles = ["paragraph", "table", "header", "figure_caption", "list_item", "footnote"]
    role_labels = {
        "paragraph":       "Paragraph Text",
        "table":           "Table Cell Content",
        "header":          "Section Header / Title",
        "figure_caption":  "Figure Caption",
        "list_item":       "List Item",
        "footnote":        "Footnote / Metadata",
    }
    
    role_colors = {
        "paragraph":       "#38bdf8",  # Sky blue
        "table":           "#f97316",  # Vibrant orange
        "header":          "#4ade80",  # Mint green
        "figure_caption":  "#f43f5e",  # Rose red
        "list_item":       "#a855f7",  # Purple
        "footnote":        "#eab308",  # Yellow
    }

    for role in unique_roles:
        mask = np.array([r == role for r in cleaned_roles])
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=role_colors[role], label=role_labels[role],
            alpha=0.65, s=8, linewidths=0,
        )

    legend = ax.legend(title="Document Element Role", title_fontsize=10, fontsize=9,
                       loc="lower right", framealpha=0.4,
                       labelcolor="white", facecolor="#1e293b", edgecolor="#475569")
    legend.get_title().set_color("white")
    
    ax.set_title("UMAP Projection of Qwen2.5-VL Node Embeddings\nGrouped by Structural Document Element Role",
                 color="white", fontsize=13, pad=14, fontweight="semibold")
    ax.tick_params(colors="#94a3b8")
    ax.set_xlabel("UMAP Dimension 1", color="#94a3b8", fontsize=10)
    ax.set_ylabel("UMAP Dimension 2", color="#94a3b8", fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    out = out_dir / f"embeddings_umap_by_node_role.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  Saved publication UMAP plot: {out}")
    return out


def plot_by_page(coords: np.ndarray, pages: list, method: str,
                 out_dir: Path):
    """Scatter plot coloured by page index (normalised 0→1 colormap)."""
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    pages_arr = np.array(pages, dtype=float)
    max_page = pages_arr.max() if pages_arr.max() > 0 else 1.0
    pages_norm = pages_arr / max_page

    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=pages_norm, cmap="plasma", alpha=0.5, s=4, linewidths=0)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Relative Page Position", color="gray")
    cbar.ax.yaxis.set_tick_params(color="gray")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="gray")

    ax.set_title(f"Qwen2.5-VL Node Embeddings — {method.upper()}\nColoured by Page Index",
                 color="white", fontsize=13, pad=12)
    ax.tick_params(colors="gray")
    ax.set_xlabel(f"{method.upper()} dim 1", color="gray")
    ax.set_ylabel(f"{method.upper()} dim 2", color="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    out = out_dir / f"embeddings_{method}_by_page.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  Saved: {out}")
    return out


def plot_by_doc(coords: np.ndarray, docs: list, method: str,
                out_dir: Path, max_docs: int = 15):
    """Scatter plot coloured by document (up to max_docs shown)."""
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    from collections import Counter
    top_docs = [d for d, _ in Counter(docs).most_common(max_docs)]
    palette = {doc: DOC_PALETTE[i % len(DOC_PALETTE)]
               for i, doc in enumerate(top_docs)}

    # Plot "other" docs first in grey
    other_mask = np.array([d not in palette for d in docs])
    if other_mask.any():
        ax.scatter(coords[other_mask, 0], coords[other_mask, 1],
                   c="#444", alpha=0.2, s=3, linewidths=0, label="_nolegend_")

    patches = []
    for doc in top_docs:
        mask = np.array([d == doc for d in docs])
        color = palette[doc]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color]*mask.sum(), alpha=0.65, s=5, linewidths=0)
        patches.append(mpatches.Patch(color=color, label=doc[:20]))

    ax.legend(handles=patches, title="Document", title_fontsize=9,
              fontsize=7, loc="lower right", framealpha=0.3,
              labelcolor="white", facecolor="#2a2a4e", edgecolor="#555",
              ncol=2)
    ax.set_title(f"Qwen2.5-VL Node Embeddings — {method.upper()}\nColoured by Document",
                 color="white", fontsize=13, pad=12)
    ax.tick_params(colors="gray")
    ax.set_xlabel(f"{method.upper()} dim 1", color="gray")
    ax.set_ylabel(f"{method.upper()} dim 2", color="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    out = out_dir / f"embeddings_{method}_by_doc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  Saved: {out}")
    return out


def run_method(method: str, embs: np.ndarray, roles, pages, docs, out_dir):
    if method == "pca":
        coords = reduce_pca(embs)
    elif method == "tsne":
        coords = reduce_tsne(embs)
    elif method == "umap":
        coords = reduce_umap(embs)
    else:
        raise ValueError(method)

    out_paths = []
    out_paths.append(plot_by_role(coords, roles, method, out_dir))
    out_paths.append(plot_by_page(coords, pages, method, out_dir))
    out_paths.append(plot_by_doc(coords, docs, method, out_dir))
    return out_paths


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graphs_dir = Path(args.graphs_dir)
    embs, roles, pages, docs = load_node_sample(
        graphs_dir, args.n_graphs, args.n_nodes, args.seed
    )

    methods = ["umap", "tsne", "pca"] if args.method == "all" else [args.method]
    all_paths = []

    for m in methods:
        logger.info(f"\n{'='*50}\n  Running {m.upper()}\n{'='*50}")
        try:
            paths = run_method(m, embs, roles, pages, docs, out_dir)
            all_paths.extend(paths)
        except Exception as e:
            logger.error(f"  {m.upper()} failed: {e}")

    print(f"\n✅  Saved {len(all_paths)} visualisation files to: {out_dir}/")
    for p in all_paths:
        print(f"   {p}")


if __name__ == "__main__":
    main()
