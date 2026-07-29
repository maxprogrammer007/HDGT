"""
edge_ablation.py

HDGT Priority 6 — Graph Edge Ablation Study

Evaluates retrieval performance with each edge type removed in turn,
measuring how much each structural relationship contributes to GNN retrieval.

Edge types tested:
  - All edges (baseline)
  - Remove 'contains'         (page→text hierarchy)
  - Remove 'reading_order'    (sequential text flow)
  - Remove 'spatial'          (proximity-based connections)
  - No edges at all           (pure node embedding, no message passing)

For each ablation, runs the GNN retrieval on `--limit` questions and
reports Recall@1, Recall@5, MRR, ELA.

Usage:
    python edge_ablation.py --limit 500
    python edge_ablation.py              # full val set (slow)
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from hdgt.evaluation.loaders import MPDocVQALoader, GraphLoader, graph_to_node_list
from hdgt.evaluation.metrics import compute_metrics

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("edge_ablation")


def parse_args():
    p = argparse.ArgumentParser(
        description="HDGT Edge Type Ablation Study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--gnn_checkpoint", default="checkpoints/hdgt_gnn_best.pt")
    p.add_argument("--top_k", type=int, default=10)
    return p.parse_args()


# Map of ablation name → rule for filtering edges
# Mode: ('keep_only', set) or ('remove', set)
ABLATION_CONFIGS = {
    "All edges (HDGT Baseline)":   ("remove", set()),
    "Spatial edges only":          ("keep_only", {"spatial"}),
    "Reading Order edges only":    ("keep_only", {"reading_order"}),
    "Contains edges only":         ("keep_only", {"contains"}),
    "Without Spatial edges":       ("remove", {"spatial"}),
    "Without Reading Order edges": ("remove", {"reading_order"}),
    "Without Contains edges":      ("remove", {"contains"}),
    "No edges (Nodes-only)":       ("remove", {"contains", "reading_order", "spatial"}),
}


def filter_edges(edge_index_dict: dict, config_rule: tuple) -> dict:
    """Filter edge_index_dict based on keep_only or remove rule."""
    mode, rels = config_rule
    if mode == "keep_only":
        return {et: ei for et, ei in edge_index_dict.items() if et[1] in rels}
    else:  # remove
        return {et: ei for et, ei in edge_index_dict.items() if et[1] not in rels}


def retrieve_with_gnn(question_emb, node_embs, edge_index_dict,
                      gnn, device, top_k):
    """Run GNN retrieval for one question over one graph."""
    x_dict = {"text": node_embs}
    with torch.no_grad():
        out_dict = gnn(x_dict, edge_index_dict)
        text_out = F.normalize(out_dict["text"], p=2, dim=-1)

        q_proj = gnn.input_projections["text"](question_emb.unsqueeze(0))
        q_out  = F.normalize(gnn.out_proj(q_proj), p=2, dim=-1)

        scores = (q_out @ text_out.T).squeeze(0)

    topk_idx = scores.topk(min(top_k, scores.shape[0])).indices.tolist()
    return topk_idx, scores


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("  HDGT Edge Ablation Study")
    print(f"{'='*60}")
    print(f"  Device   : {device}")
    print(f"  Limit    : {args.limit}")
    print(f"  Top-K    : {args.top_k}")
    print(f"{'='*60}\n")

    # Load models
    from hdgt.models import HDGTHeteroGNN
    gnn = HDGTHeteroGNN(in_dim=2048, hidden_dim=256, out_dim=128).to(device)
    ckpt = Path(args.gnn_checkpoint)
    if ckpt.exists():
        gnn.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f"  ✅ GNN loaded from {ckpt}")
    else:
        print(f"  ⚠️  No checkpoint — using random weights")
    gnn.eval()

    # Load QwenVLEmbedder for query embedding
    from hdgt.models import QwenVLEmbedder
    logger.info("Loading QwenVLEmbedder...")
    embedder = QwenVLEmbedder(device=str(device))

    def embed_query(q):
        with torch.no_grad():
            emb = embedder.embed_query(q).to(device) # [1, 2048]
        return emb.squeeze(0).float()

    # Data
    qa_loader  = MPDocVQALoader(data_root="data/MP-DocVQA", split="val")
    graph_loader = GraphLoader(graphs_dir="experiments/mpdocvqa")
    limit = args.limit

    # Accumulators: one dict per ablation config
    metrics_acc = {name: {"n": 0, "recall@1": 0.0, "recall@5": 0.0,
                          "recall@10": 0.0, "mrr": 0.0, "ela": 0.0}
                   for name in ABLATION_CONFIGS}

    evaluated_count = 0
    for idx, item in enumerate(tqdm(qa_loader, desc="Edge Ablation", total=limit or len(qa_loader))):
        if (limit is not None and idx >= limit) or evaluated_count >= 3472:
            break

        if idx % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        question   = item["question"]
        page_ids   = item["page_ids"]
        gt_page    = item.get("answer_page_idx", 0)
        gt_answers = item.get("answers", [])
        ctx_id     = qa_loader.build_context_id(page_ids)

        g = graph_loader.load(ctx_id)
        if g is None or "text" not in g.node_types:
            continue
        evaluated_count += 1
        main._missing_count = 0
        txt = g["text"]
        if not hasattr(txt, "qwen_x") or txt.qwen_x is None:
            continue

        node_list  = graph_to_node_list(g)
        text_nodes = [n for n in node_list if n["type"] == "text"]
        node_embs  = txt.qwen_x.to(device).float()

        if node_embs.shape[0] != len(text_nodes):
            continue

        # Build full edge_index_dict
        full_edge_dict = {}
        for et in g.edge_types:
            if hasattr(g[et], "edge_index") and g[et].edge_index.numel() > 0:
                full_edge_dict[et] = g[et].edge_index.to(device)

        # Query embedding (computed once per question)
        try:
            q_emb = embed_query(question)
            # Resize to GNN input dim (2048) if needed
            if q_emb.shape[0] != 2048:
                q_emb = F.adaptive_avg_pool1d(
                    q_emb.unsqueeze(0).unsqueeze(0), 2048
                ).squeeze()
        except Exception:
            continue

        # Run each ablation
        for config_name, rule in ABLATION_CONFIGS.items():
            edge_dict = filter_edges(full_edge_dict, rule)
            try:
                topk_idx, scores = retrieve_with_gnn(
                    q_emb, node_embs, edge_dict, gnn, device, args.top_k
                )
                top_nodes = [text_nodes[i] for i in topk_idx]
                m = compute_metrics(
                    retrieved_pages   = [n["page"]    for n in top_nodes],
                    retrieved_contents= [n.get("content","") for n in top_nodes],
                    ground_truth_page = gt_page,
                    ground_truth_answers = gt_answers,
                )
                acc = metrics_acc[config_name]
                for k in ["recall@1","recall@5","recall@10","mrr","ela"]:
                    acc[k] += m.get(k, 0.0)
                acc["n"] += 1
            except Exception:
                pass

    # Print results table
    metric_keys = ["recall@1", "recall@5", "recall@10", "mrr", "ela"]
    print(f"\n{'='*80}")
    print("  EDGE ABLATION RESULTS — Δ shows change vs. 'All edges' baseline")
    print(f"{'='*80}")
    header = f"  {'Configuration':<35}" + "".join(f" {m.upper():>9}" for m in metric_keys)
    print(header)
    print("  " + "-" * (len(header) - 2))

    baseline = None
    for name, acc in metrics_acc.items():
        n = acc["n"]
        if n == 0:
            print(f"  {name:<35}  (no data)")
            continue
        vals = {k: acc[k] / n for k in metric_keys}
        if baseline is None:
            baseline = vals
            row = f"  {name:<35}"
            for k in metric_keys:
                v = vals[k]
                row += f"  {v*100:6.2f}%  " if "recall" in k else f"  {v:.4f}  "
            print(row)
        else:
            row = f"  {name:<35}"
            for k in metric_keys:
                v = vals[k]
                delta = v - baseline[k]
                sign = "+" if delta >= 0 else ""
                if "recall" in k:
                    row += f"  {v*100:5.2f}% ({sign}{delta*100:.1f})"
                else:
                    row += f"  {v:.4f}({sign}{delta:.4f})"
            print(row)

    print(f"{'='*80}\n")

    # Save JSON
    out = Path("experiments/edge_ablation_results.json")
    results = {}
    for name, acc in metrics_acc.items():
        n = acc["n"]
        results[name] = {
            "n": n,
            **({k: acc[k] / n for k in metric_keys} if n > 0 else {}),
        }
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅  Results saved to: {out}")


if __name__ == "__main__":
    main()
