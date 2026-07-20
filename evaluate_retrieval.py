"""
evaluate_retrieval.py

Phase 1.6 — CLI script for running the HDGT retrieval evaluation pipeline
on the MP-DocVQA dataset.

Usage
-----
# Full val-set BM25 evaluation
python evaluate_retrieval.py --split val --method bm25

# Quick smoke-test (first 20 questions, mock graph traversal)
python evaluate_retrieval.py --split val --method mock-graph --limit 20

# Random baseline for comparison
python evaluate_retrieval.py --split val --method random --limit 100 --k 5

Results are saved to:
    experiments/retrieval_results_{split}_{method}.jsonl

A summary table is printed to stdout at the end.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from hdgt.evaluation.loaders import MPDocVQALoader, GraphLoader
from hdgt.evaluation.metrics import compute_metrics
from hdgt.evaluation.retriever import get_retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("evaluate_retrieval")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HDGT Phase 1.6 — Retrieval Evaluation on MP-DocVQA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--method",
        choices=["random", "bm25", "mock-graph"],
        default="bm25",
        help="Retrieval method to use.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Primary top-K cutoff for Recall@K display. "
             "Recall@1, @5, @10 are always computed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to the first N questions. "
             "Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--data-root",
        default="data/MP-DocVQA",
        help="Path to the MP-DocVQA data directory (must contain qas.zip).",
    )
    parser.add_argument(
        "--graphs-dir",
        default="experiments/mpdocvqa",
        help="Path to directory containing compiled *_graph.pt files.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments",
        help="Directory where retrieval_results_*.jsonl is saved.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print a progress update every N questions.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("  HDGT Phase 1.6 — Retrieval Evaluation")
    print("=" * 60)
    print(f"  Split   : {args.split}")
    print(f"  Method  : {args.method}")
    print(f"  Limit   : {args.limit or 'All'}")
    print(f"  Graphs  : {args.graphs_dir}")
    print("=" * 60)

    # ── Initialise components ───────────────────────────────────────────
    qa_loader    = MPDocVQALoader(args.data_root, split=args.split, limit=args.limit)
    graph_loader = GraphLoader(args.graphs_dir)
    retriever    = get_retriever(args.method)

    total          = len(qa_loader)
    skipped_no_graph = 0

    # Running metric accumulators
    sum_r1  = sum_r5  = sum_r10 = 0.0
    sum_mrr = sum_ela = sum_anls = 0.0
    evaluated = 0

    # Output file
    output_path = Path(args.output_dir) / f"retrieval_results_{args.split}_{args.method}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:

        for idx, qa in enumerate(qa_loader):

            # ── Derive context_id and load graph ───────────────────────
            context_id = qa_loader.build_context_id(qa["page_ids"])
            graph      = graph_loader.load(context_id)

            if graph is None:
                skipped_no_graph += 1
                if skipped_no_graph <= 5:
                    logger.warning(
                        f"No graph for context '{context_id}' "
                        f"(q_id={qa['question_id']}). "
                        f"Run prepare_mpdocvqa.py + build_mpdocvqa_graphs.py first."
                    )
                continue

            # ── Run retrieval ──────────────────────────────────────────
            try:
                results = retriever.retrieve(qa["question"], graph, top_k=10)
            except Exception as e:
                logger.warning(
                    f"Retriever failed for q_id={qa['question_id']}: {e}"
                )
                continue

            retrieved_pages    = [r["page"] for r in results]
            retrieved_contents = [r["content"] for r in results]
            gt_page            = qa["answer_page_idx"]  # 0-indexed within context
            gt_answers         = qa["answers"]

            # ── Compute metrics ────────────────────────────────────────
            metrics = compute_metrics(
                retrieved_pages=retrieved_pages,
                retrieved_contents=retrieved_contents,
                ground_truth_page=gt_page if gt_page is not None else -1,
                ground_truth_answers=gt_answers,
            )

            sum_r1   += metrics["recall@1"]
            sum_r5   += metrics["recall@5"]
            sum_r10  += metrics["recall@10"]
            sum_mrr  += metrics["mrr"]
            sum_ela  += metrics["ela"]
            sum_anls += metrics["anls"]
            evaluated += 1

            # ── Write per-question result ──────────────────────────────
            record = {
                "question_id":      qa["question_id"],
                "query":            qa["question"],
                "context_id":       context_id,
                "ground_truth_page": gt_page,
                "ground_truth_answers": gt_answers,
                "retrieved_nodes": [
                    {
                        "node_uid": r.get("node_uid", ""),
                        "type":     r.get("type", ""),
                        "page":     r.get("page", -1),
                        "rank":     rank + 1,
                        "score":    round(r.get("score", 0.0), 6),
                        "content":  r.get("content", "")[:300],  # truncate for file size
                        "bbox":     r.get("bbox", []),
                    }
                    for rank, r in enumerate(results)
                ],
                "metrics": {k: round(v, 4) for k, v in metrics.items()},
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # ── Progress logging ───────────────────────────────────────
            if (idx + 1) % args.log_every == 0:
                n = evaluated
                logger.info(
                    f"[{idx+1}/{total}] evaluated={n} | "
                    f"R@1={sum_r1/n:.3f} R@5={sum_r5/n:.3f} "
                    f"MRR={sum_mrr/n:.3f} ELA={sum_ela/n:.3f}"
                )

    # ── Final summary ──────────────────────────────────────────────────
    n = max(evaluated, 1)
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Method         : {args.method}")
    print(f"  Split          : {args.split}")
    print(f"  Evaluated      : {evaluated} / {total} questions")
    if skipped_no_graph > 0:
        print(f"  Skipped (no graph): {skipped_no_graph}")
    print("-" * 60)
    print(f"  Recall@1       : {sum_r1 / n:.4f}")
    print(f"  Recall@5       : {sum_r5 / n:.4f}")
    print(f"  Recall@10      : {sum_r10 / n:.4f}")
    print(f"  MRR            : {sum_mrr / n:.4f}")
    print(f"  ELA            : {sum_ela / n:.4f}")
    print(f"  ANLS (proxy)   : {sum_anls / n:.4f}")
    print("=" * 60)
    print(f"  Results saved  : {output_path}")
    print("=" * 60)

    if evaluated == 0:
        print(
            "\n[WARNING] No questions were evaluated.\n"
            "This is expected if graph files have not been compiled yet.\n"
            "On the workstation, run:\n"
            "  1. python prepare_mpdocvqa.py\n"
            "  2. python build_mpdocvqa_graphs.py\n"
            "  3. python evaluate_retrieval.py --split val --method bm25\n"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    evaluate(parse_args())
