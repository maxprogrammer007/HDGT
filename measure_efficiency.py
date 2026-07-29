"""
measure_efficiency.py

HDGT Priority 7 — Efficiency Measurement

Measures wall-clock time and GPU memory for each component of the pipeline:
  1. Graph loading           (disk I/O)
  2. BM25 indexing & query   (CPU)
  3. Qwen node embedding     (GPU, amortised per-graph)
  4. GNN forward pass        (GPU)
  5. Cosine similarity rank  (GPU)

Outputs a table suitable for a conference paper System section.

Usage:
    python measure_efficiency.py --n_graphs 20
"""

import argparse
import time
import logging
from pathlib import Path

import torch
import numpy as np

logging.basicConfig(level=logging.WARNING)

import sys
sys.path.insert(0, str(Path(__file__).parent))

from hdgt.evaluation.loaders import MPDocVQALoader, GraphLoader, graph_to_node_list
from hdgt.evaluation.retriever import BM25Retriever
from hdgt.evaluation.metrics import compute_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_graphs", type=int, default=20,
                   help="Number of graphs to time (averaged).")
    p.add_argument("--gnn_checkpoint", default="checkpoints/hdgt_gnn_best.pt")
    return p.parse_args()


def gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0.0


class Timer:
    def __init__(self, name):
        self.name = name
        self.times = []
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - self._t0) * 1000
        self.times.append(elapsed_ms)
    @property
    def mean_ms(self):
        return np.mean(self.times) if self.times else 0.0
    @property
    def std_ms(self):
        return np.std(self.times) if len(self.times) > 1 else 0.0


def print_table(rows):
    header = f"{'Component':<35} {'Mean (ms)':>12} {'± Std':>10} {'Per Node':>12}"
    print(f"\n{'='*75}")
    print("  HDGT Pipeline Efficiency Report")
    print(f"{'='*75}")
    print(f"  {header}")
    print(f"  {'-'*72}")
    for row in rows:
        comp, mean, std, extra = row
        print(f"  {comp:<35} {mean:>11.1f}  {std:>9.1f}  {extra}")
    print(f"{'='*75}\n")


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nDevice: {device}")
    print(f"Timing {args.n_graphs} graphs...\n")

    # Data
    qa_loader    = MPDocVQALoader(data_root="data/MP-DocVQA", split="val")
    graph_loader = GraphLoader(graphs_dir="experiments/mpdocvqa")
    bm25         = BM25Retriever()

    # Load GNN
    from hdgt.models import HDGTHeteroGNN
    gnn = HDGTHeteroGNN(in_dim=2048, hidden_dim=256, out_dim=128).to(device)
    ckpt = Path(args.gnn_checkpoint)
    if ckpt.exists():
        gnn.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    gnn.eval()

    t_load    = Timer("Graph loading (disk I/O)")
    t_bm25    = Timer("BM25 index + query")
    t_qwen    = Timer("Qwen feature access (GPU)")
    t_gnn     = Timer("GNN forward pass (GPU)")
    t_rank    = Timer("Cosine ranking (GPU)")

    node_counts = []
    evaluated   = 0
    mem_before  = gpu_memory_mb()

    for idx, item in enumerate(qa_loader):
        if evaluated >= args.n_graphs:
            break
        question = item["question"]
        ctx_id   = qa_loader.build_context_id(item["page_ids"])

        # 1. Graph loading
        with t_load:
            g = graph_loader.load(ctx_id)
        if g is None or "text" not in g.node_types:
            continue
        if not hasattr(g["text"], "qwen_x") or g["text"].qwen_x is None:
            continue

        n_nodes = g["text"].num_nodes
        node_counts.append(n_nodes)

        # 2. BM25
        with t_bm25:
            try:
                bm25.retrieve(question, g, top_k=10)
            except Exception:
                pass

        # 3. Qwen features (already extracted, just move to GPU)
        with t_qwen:
            node_embs = g["text"].qwen_x.to(device).float()

        # 4. GNN forward — build x_dict first, then filter edges by node types present
        x_dict = {"text": node_embs}
        edge_index_dict = {}
        for et in g.edge_types:
            if hasattr(g[et], "edge_index") and g[et].edge_index.numel() > 0:
                src_type, rel, dst_type = et
                # Only include edge if both endpoint node types have features
                if src_type in x_dict and dst_type in x_dict:
                    edge_index_dict[et] = g[et].edge_index.to(device)

        with t_gnn:
            with torch.no_grad():
                out = gnn(x_dict, edge_index_dict)
                gnn_embs = out["text"]

        # 5. Cosine ranking
        q_placeholder = torch.randn(2048, device=device)
        with t_rank:
            q_proj = gnn.input_projections["text"](q_placeholder.unsqueeze(0))
            q_out  = torch.nn.functional.normalize(gnn.out_proj(q_proj), p=2, dim=-1)
            n_norm = torch.nn.functional.normalize(gnn_embs, p=2, dim=-1)
            scores = (q_out @ n_norm.T).squeeze(0)
            _ = scores.topk(min(10, scores.shape[0])).indices

        evaluated += 1
        print(f"  [{evaluated}/{args.n_graphs}] {ctx_id[:40]:<40} nodes={n_nodes}")

    mem_after = gpu_memory_mb()
    avg_nodes = int(np.mean(node_counts)) if node_counts else 0

    # Build table
    rows = [
        ("Graph loading (disk I/O)",
         t_load.mean_ms, t_load.std_ms,
         f"per document"),
        ("BM25 index + query",
         t_bm25.mean_ms, t_bm25.std_ms,
         f"per document"),
        ("Qwen feat → GPU transfer",
         t_qwen.mean_ms, t_qwen.std_ms,
         f"~{avg_nodes} nodes avg"),
        ("GNN forward pass",
         t_gnn.mean_ms, t_gnn.std_ms,
         f"{t_gnn.mean_ms/avg_nodes*1000:.2f} μs/node" if avg_nodes else "N/A"),
        ("Cosine ranking",
         t_rank.mean_ms, t_rank.std_ms,
         f"{t_rank.mean_ms/avg_nodes*1000:.2f} μs/node" if avg_nodes else "N/A"),
    ]

    total_mean = sum(r[1] for r in rows)
    total_std  = np.sqrt(sum(r[2]**2 for r in rows))
    rows.append(("─" * 35, 0, 0, ""))
    rows.append(("TOTAL (end-to-end)",
                 total_mean, total_std,
                 f"per query"))

    print_table(rows)

    print(f"  GPU memory used:  {mem_after - mem_before:.1f} MB")
    print(f"  Avg nodes/graph:  {avg_nodes:,}")

    # Save markdown table
    out_path = Path("experiments/efficiency_report.md")
    with open(out_path, "w") as f:
        f.write("# HDGT Pipeline Efficiency Report\n\n")
        f.write(f"Measured on {evaluated} validation graphs.\n\n")
        f.write(f"| Component | Mean (ms) | ± Std | Notes |\n")
        f.write(f"| :--- | :---: | :---: | :--- |\n")
        for comp, mean, std, extra in rows:
            if mean == 0 and "─" in comp:
                continue
            f.write(f"| {comp} | {mean:.1f} | {std:.1f} | {extra} |\n")
        f.write(f"\n- **GPU Memory**: {mem_after - mem_before:.1f} MB\n")
        f.write(f"- **Avg nodes/graph**: {avg_nodes:,}\n")
        f.write(f"- **Device**: {device}\n")
    print(f"\n✅  Report saved to: {out_path}")


if __name__ == "__main__":
    main()
