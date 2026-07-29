"""
compare_gnn_baselines.py

HDGT Priority 10 — GNN Baseline Comparison

Compares heterogeneous graph message passing (HDGT HeteroGNN) against a
homogeneous GraphSAGE baseline (ignoring edge/node types) on a sampled
subset of 300 validation questions.

Architectures evaluated:
  1. Homogeneous GraphSAGE (SAGEConv over flattened graph, ignoring edge types)
  2. HDGT Heterogeneous GNN (HeteroConv with edge-type specific message passing)

Usage:
    python compare_gnn_baselines.py --limit 300
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from hdgt.evaluation.loaders import MPDocVQALoader, GraphLoader, graph_to_node_list
from hdgt.evaluation.metrics import compute_metrics
from hdgt.models import QwenVLEmbedder, HDGTHeteroGNN

logging.basicConfig(level=logging.WARNING)


class HomogeneousGraphSAGE(nn.Module):
    """Homogeneous GraphSAGE baseline ignoring edge types."""
    def __init__(self, in_dim=2048, hidden_dim=256, out_dim=128):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.proj(x))
        h = F.relu(self.conv1(h, edge_index))
        h = self.conv2(h, edge_index)
        return F.normalize(h, p=2, dim=-1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top_k", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("  HDGT vs. Homogeneous GraphSAGE Baseline Comparison")
    print(f"{'='*70}")
    print(f"  Device : {device}")
    print(f"  Limit  : {args.limit}")
    print(f"{'='*70}\n")

    # Load Qwen embedder
    embedder = QwenVLEmbedder(device=str(device))

    # Load HDGT HeteroGNN
    hdgt_gnn = HDGTHeteroGNN(in_dim=2048, hidden_dim=256, out_dim=128).to(device)
    ckpt = Path("checkpoints/hdgt_gnn_best.pt")
    if ckpt.exists():
        hdgt_gnn.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    hdgt_gnn.eval()

    # Instantiate Homogeneous GraphSAGE baseline
    sage_baseline = HomogeneousGraphSAGE(in_dim=2048, hidden_dim=256, out_dim=128).to(device)
    sage_baseline.eval()

    qa_loader    = MPDocVQALoader(data_root="data/MP-DocVQA", split="val")
    graph_loader = GraphLoader(graphs_dir="experiments/mpdocvqa")
    limit        = args.limit

    acc_hdgt = {"n": 0, "recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0, "ela": 0.0}
    acc_sage = {"n": 0, "recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0, "ela": 0.0}

    evaluated_count = 0
    for idx, item in enumerate(tqdm(qa_loader, desc="GNN Baseline Comparison", total=limit or len(qa_loader))):
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

        # Embed question query
        with torch.no_grad():
            q_raw = embedder.embed_query(question).to(device)

        # 1. HDGT Heterogeneous GNN evaluation
        try:
            x_dict = {"text": node_embs}
            edge_index_dict = {}
            for et in g.edge_types:
                if hasattr(g[et], "edge_index") and g[et].edge_index.numel() > 0:
                    src_type, rel, dst_type = et
                    if src_type in x_dict and dst_type in x_dict:
                        edge_index_dict[et] = g[et].edge_index.to(device)

            with torch.no_grad():
                out_h = hdgt_gnn(x_dict, edge_index_dict)
                text_h = F.normalize(out_h["text"], p=2, dim=-1)

                q_proj = hdgt_gnn.input_projections["text"](q_raw)
                q_out  = F.normalize(hdgt_gnn.out_proj(q_proj), p=2, dim=-1)

                scores = (q_out @ text_h.T).squeeze(0)

            topk_idx = scores.topk(min(args.top_k, scores.shape[0])).indices.tolist()
            top_nodes = [text_nodes[i] for i in topk_idx]
            m = compute_metrics(
                retrieved_pages=[n["page"] for n in top_nodes],
                retrieved_contents=[n.get("content","") for n in top_nodes],
                ground_truth_page=gt_page,
                ground_truth_answers=gt_answers,
            )
            for k in ["recall@1","recall@5","recall@10","mrr","ela"]:
                acc_hdgt[k] += m.get(k, 0.0)
            acc_hdgt["n"] += 1
        except Exception:
            pass

        # 2. Homogeneous GraphSAGE baseline evaluation
        try:
            # Flatten all edge indices into one unified edge_index tensor
            all_edges = []
            for et in g.edge_types:
                if hasattr(g[et], "edge_index") and g[et].edge_index.numel() > 0:
                    src_type, rel, dst_type = et
                    if src_type == "text" and dst_type == "text":
                        all_edges.append(g[et].edge_index.to(device))
            if all_edges:
                flat_edge_index = torch.cat(all_edges, dim=1)
            else:
                flat_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

            with torch.no_grad():
                sage_embs = sage_baseline(node_embs, flat_edge_index)
                q_proj_sage = F.adaptive_avg_pool1d(q_raw.unsqueeze(0), 128).squeeze(0)
                q_proj_sage = F.normalize(q_proj_sage, p=2, dim=-1)
                sage_scores = (q_proj_sage @ sage_embs.T).squeeze(0)

            topk_idx_s = sage_scores.topk(min(args.top_k, sage_scores.shape[0])).indices.tolist()
            top_nodes_s = [text_nodes[i] for i in topk_idx_s]
            m_s = compute_metrics(
                retrieved_pages=[n["page"] for n in top_nodes_s],
                retrieved_contents=[n.get("content","") for n in top_nodes_s],
                ground_truth_page=gt_page,
                ground_truth_answers=gt_answers,
            )
            for k in ["recall@1","recall@5","recall@10","mrr","ela"]:
                acc_sage[k] += m_s.get(k, 0.0)
            acc_sage["n"] += 1
        except Exception:
            pass

    # Print results
    metric_keys = ["recall@1", "recall@5", "recall@10", "mrr", "ela"]
    print(f"\n{'='*75}")
    print("  GNN ARCHITECTURE COMPARISON — MP-DocVQA Validation Subset")
    print(f"{'='*75}")
    header = f"  {'Model Architecture':<32}" + "".join(f" {m.upper():>9}" for m in metric_keys) + "   N"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name, acc in [("Homogeneous GraphSAGE", acc_sage), ("HDGT Heterogeneous GNN", acc_hdgt)]:
        n = acc["n"]
        if n == 0:
            print(f"  {name:<32}  (no data)")
            continue
        row = f"  {name:<32}"
        for k in metric_keys:
            v = acc[k] / n
            row += f"  {v*100:6.2f}%  " if "recall" in k else f"  {v:.4f}  "
        row += f"  {n}"
        print(row)
    print(f"{'='*75}\n")

    # Save JSON
    out_file = Path("experiments/gnn_baseline_comparison.json")
    res = {
        "Homogeneous GraphSAGE": {k: acc_sage[k]/max(1, acc_sage["n"]) for k in metric_keys},
        "HDGT Heterogeneous GNN": {k: acc_hdgt[k]/max(1, acc_hdgt["n"]) for k in metric_keys},
    }
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"✅  Results saved to: {out_file}")


if __name__ == "__main__":
    main()
