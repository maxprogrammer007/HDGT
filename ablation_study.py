"""
ablation_study.py

HDGT Ablation Study — Priority 3 from research review.

Runs multiple retrieval configurations and reports a comparison table:
  1. BM25 only               — lexical keyword baseline
  2. Qwen embeddings only     — cosine sim on Qwen2.5-VL features (no GNN)
  3. Qwen + HDGT GNN          — GNN-enhanced retrieval
  4. BM25 + Qwen (oracle)     — BM25 stage1 + cosine rerank (no GNN needed)

This isolates the contribution of each component clearly.

Usage:
    python ablation_study.py --limit 500  # quick run
    python ablation_study.py              # full val set
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from hdgt.evaluation.loaders import MPDocVQALoader, GraphLoader, graph_to_node_list
from hdgt.evaluation.retriever import BM25Retriever
from hdgt.evaluation.metrics import compute_metrics

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("ablation_study")


def parse_args():
    p = argparse.ArgumentParser(
        description="HDGT Ablation Study — Component Contribution Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--limit", type=int, default=None,
                   help="Limit to first N questions (None = full val set).")
    p.add_argument("--gnn_checkpoint", type=str, default="checkpoints/hdgt_gnn_best.pt")
    p.add_argument("--top_k", type=int, default=10)
    return p.parse_args()


def embed_query_qwen(question: str, embedder, device) -> torch.Tensor:
    """Extract Qwen2.5-VL text embedding for a question string."""
    with torch.no_grad():
        emb = embedder.embed_query(question).to(device)
    return emb.squeeze(0).float()


def cosine_retrieval(q_emb: torch.Tensor, node_embs: torch.Tensor,
                     nodes: list, top_k: int) -> list:
    """Retrieve top_k nodes by cosine similarity."""
    q_norm = F.normalize(q_emb.unsqueeze(0), p=2, dim=-1)
    n_norm = F.normalize(node_embs, p=2, dim=-1)
    scores = (q_norm @ n_norm.T).squeeze(0)  # [N]
    topk = scores.topk(min(top_k, len(nodes))).indices.tolist()
    return [
        {**nodes[i], "score": float(scores[i])}
        for i in topk
    ]


def accumulate_metrics(results: list) -> dict:
    """Average metrics across all evaluated questions."""
    if not results:
        return {}
    keys = [k for k in results[0].keys() if k != "anls"]
    return {k: sum(r[k] for r in results) / len(results) for k in keys}


def print_table(configs: dict[str, dict]):
    """Print a formatted comparison table."""
    metrics = ["recall@1", "recall@5", "recall@10", "mrr", "ela"]
    hdr = f"{'Model':<30} " + " ".join(f"{m.upper():>12}" for m in metrics) + "  N"
    print()
    print("=" * (len(hdr) + 4))
    print("  ABLATION STUDY RESULTS — MP-DocVQA Validation Set")
    print("=" * (len(hdr) + 4))
    print(f"  {hdr}")
    print("  " + "-" * len(hdr))
    for name, data in configs.items():
        n = data.get("n", 0)
        if n == 0:
            row = f"  {name:<30} " + " ".join(f"{'N/A':>12}" for _ in metrics) + f"  {n}"
        else:
            row = f"  {name:<30} "
            for m in metrics:
                v = data.get(m, 0.0)
                if m == "ela":
                    row += f"  {v:.4f}    "
                elif m == "mrr":
                    row += f"  {v:.4f}    "
                else:
                    row += f"  {v*100:6.2f}%    "
            row += f"  {n}"
        print(row)
    print("=" * (len(hdr) + 4))
    print()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'='*65}")
    print("  HDGT Ablation Study — Component Contribution Analysis")
    print(f"{'='*65}")
    print(f"  Device: {device}")
    print(f"  Limit:  {args.limit or 'Full val set'}")
    print(f"{'='*65}\n")

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    qa_loader = MPDocVQALoader(data_root="data/MP-DocVQA", split="val")
    graph_loader = GraphLoader(graphs_dir="experiments/mpdocvqa")
    bm25 = BM25Retriever()
    limit = args.limit or len(qa_loader)

    # ------------------------------------------------------------------ #
    # Load Qwen2.5-VL for text-only embedding
    # ------------------------------------------------------------------ #
    print("Loading Qwen2.5-VL Embedder for query embedding...")
    try:
        from hdgt.models import QwenVLEmbedder
        q_embedder = QwenVLEmbedder(device=str(device))
        qwen_available = True
        print("  ✅ Qwen2.5-VL loaded")
    except Exception as e:
        print(f"  ⚠️  Qwen2.5-VL not available: {e}")
        qwen_available = False
        q_embedder = None

    # ------------------------------------------------------------------ #
    # Load HDGT GNN
    # ------------------------------------------------------------------ #
    gnn = None
    try:
        from hdgt.models import HDGTHeteroGNN
        gnn = HDGTHeteroGNN(in_dim=2048, hidden_dim=256, out_dim=128).to(device)
        ckpt = Path(args.gnn_checkpoint)
        if ckpt.exists():
            gnn.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            gnn.eval()
            gnn_available = True
            print("  ✅ HDGT GNN loaded from checkpoint")
        else:
            gnn_available = False
            print(f"  ⚠️  No checkpoint at {ckpt}")
    except Exception as e:
        gnn_available = False
        print(f"  ⚠️  GNN not available: {e}")

    # ------------------------------------------------------------------ #
    # Storage for each configuration
    # ------------------------------------------------------------------ #
    configs = {
        "BM25 only":             {"results": [], "n": 0},
        "Qwen emb only":         {"results": [], "n": 0},
        "BM25 + Qwen (rerank)":  {"results": [], "n": 0},
        "Qwen + HDGT GNN":       {"results": [], "n": 0},
    }

    # ------------------------------------------------------------------ #
    # Main evaluation loop
    # ------------------------------------------------------------------ #
    out_dir = Path("experiments")
    out_dir.mkdir(exist_ok=True)

    evaluated_count = 0
    for idx, item in enumerate(tqdm(qa_loader, desc="Ablation", total=limit)):
        if idx >= limit or evaluated_count >= 3472:
            break

        if idx % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        question     = item["question"]
        page_ids     = item["page_ids"]
        gt_page      = item.get("answer_page_idx", 0)
        gt_answers   = item.get("answers", [])
        context_id   = qa_loader.build_context_id(page_ids)

        g = graph_loader.load(context_id)
        if g is None:
            continue
        evaluated_count += 1

        # ----- BM25 only -----
        try:
            bm25_nodes = bm25.retrieve(question, g, top_k=args.top_k)
            m = compute_metrics(
                retrieved_pages=[n["page"] for n in bm25_nodes],
                retrieved_contents=[n.get("content","") for n in bm25_nodes],
                ground_truth_page=gt_page,
                ground_truth_answers=gt_answers,
            )
            configs["BM25 only"]["results"].append(m)
            configs["BM25 only"]["n"] += 1
        except Exception:
            pass

        if not qwen_available:
            continue

        has_qwen = (
            "text" in g.node_types
            and hasattr(g["text"], "qwen_x")
            and g["text"].qwen_x is not None
        )
        if not has_qwen:
            continue

        node_list = graph_to_node_list(g)
        text_nodes = [n for n in node_list if n["type"] == "text"]
        if not text_nodes:
            continue

        # Build node embedding matrix
        node_embs = g["text"].qwen_x.to(device)  # [N, 2048]
        if node_embs.shape[0] != len(text_nodes):
            continue

        # Query embedding via Qwen hidden states
        try:
            q_emb = embed_query_qwen(question, q_embedder, device)
            # Resize q_emb to match node dim if needed
            if q_emb.shape[0] != node_embs.shape[1]:
                q_emb = F.adaptive_avg_pool1d(
                    q_emb.unsqueeze(0).unsqueeze(0),
                    node_embs.shape[1]
                ).squeeze()
        except Exception:
            continue

        # ----- Qwen only -----
        try:
            top_nodes = cosine_retrieval(q_emb, node_embs, text_nodes, args.top_k)
            m = compute_metrics(
                retrieved_pages=[n["page"] for n in top_nodes],
                retrieved_contents=[n.get("content","") for n in top_nodes],
                ground_truth_page=gt_page,
                ground_truth_answers=gt_answers,
            )
            configs["Qwen emb only"]["results"].append(m)
            configs["Qwen emb only"]["n"] += 1
        except Exception:
            pass

        # ----- BM25 + Qwen rerank -----
        try:
            bm25_top = bm25.retrieve(question, g, top_k=5)
            bm25_pages = {n["page"] for n in bm25_top}
            # Only rerank text nodes on BM25-retrieved pages
            candidate_idxs = [
                i for i, n in enumerate(text_nodes)
                if n["page"] in bm25_pages
            ]
            if candidate_idxs:
                cand_embs = node_embs[candidate_idxs]
                cand_nodes = [text_nodes[i] for i in candidate_idxs]
                top_nodes = cosine_retrieval(q_emb, cand_embs, cand_nodes, args.top_k)
                m = compute_metrics(
                    retrieved_pages=[n["page"] for n in top_nodes],
                    retrieved_contents=[n.get("content","") for n in top_nodes],
                    ground_truth_page=gt_page,
                    ground_truth_answers=gt_answers,
                )
                configs["BM25 + Qwen (rerank)"]["results"].append(m)
                configs["BM25 + Qwen (rerank)"]["n"] += 1
        except Exception:
            pass

        # ----- Qwen + HDGT GNN -----
        if gnn_available:
            try:
                x_dict = {"text": node_embs}
                for nt in g.node_types:
                    if nt != "text" and hasattr(g[nt], "num_nodes") and g[nt].num_nodes > 0:
                        x_dict[nt] = torch.zeros((g[nt].num_nodes, 2048), device=device)
                edge_index_dict = {}
                for et in g.edge_types:
                    if hasattr(g[et], "edge_index") and g[et].edge_index.numel() > 0:
                        edge_index_dict[et] = g[et].edge_index.to(device)

                with torch.no_grad():
                    out_dict = gnn(x_dict, edge_index_dict)
                    gnn_embs = out_dict["text"]  # [N, 128]

                # Project query similarly
                q_gnn = gnn.input_projections["text"](q_emb.unsqueeze(0))
                q_gnn = F.normalize(gnn.out_proj(q_gnn), p=2, dim=-1)
                gnn_embs_norm = F.normalize(gnn_embs, p=2, dim=-1)

                scores = (q_gnn @ gnn_embs_norm.T).squeeze(0).cpu()
                topk_idx = scores.topk(min(args.top_k, len(text_nodes))).indices.tolist()
                top_nodes = [{**text_nodes[i], "score": float(scores[i])} for i in topk_idx]

                m = compute_metrics(
                    retrieved_pages=[n["page"] for n in top_nodes],
                    retrieved_contents=[n.get("content","") for n in top_nodes],
                    ground_truth_page=gt_page,
                    ground_truth_answers=gt_answers,
                )
                configs["Qwen + HDGT GNN"]["results"].append(m)
                configs["Qwen + HDGT GNN"]["n"] += 1
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Aggregate and print results
    # ------------------------------------------------------------------ #
    summary = {}
    all_metrics = ["recall@1", "recall@5", "recall@10", "mrr", "ela"]
    for name, data in configs.items():
        if data["results"]:
            avg = accumulate_metrics(data["results"])
            summary[name] = {**avg, "n": data["n"]}
        else:
            summary[name] = {"n": 0}

    print_table(summary)

    # Save JSON
    out_path = Path("experiments/ablation_study_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
