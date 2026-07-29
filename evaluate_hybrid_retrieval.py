"""
evaluate_hybrid_retrieval.py

HDGT Phase 3 — Two-Stage Hybrid Retrieval:
    Stage 1 : BM25  (document/page-level recall, high coverage)
    Stage 2 : HDGT Heterogeneous GNN reranker (evidence localization)

Scoring:
    score(q, node) = λ * BM25_page_score(q, page)
                   + (1-λ) * cos(q_emb, node_emb)
                   + α * GNN_score(node)

This design exploits the complementary strengths observed in Phase 2:
- BM25 is better at finding the right document/page (keyword recall)
- Qwen2.5-VL + GNN is better at locating the answer region (ELA)

Usage:
    python evaluate_hybrid_retrieval.py
    python evaluate_hybrid_retrieval.py --lambda_bm25 0.6 --top_k_bm25 10
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from hdgt.models import QwenVLEmbedder, HDGTHeteroGNN
from hdgt.evaluation.loaders import MPDocVQALoader, GraphLoader
from hdgt.evaluation.retriever import BM25Retriever
from hdgt.evaluation.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("evaluate_hybrid_retrieval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HDGT Phase 3 — Two-Stage Hybrid Retrieval Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lambda_bm25", type=float, default=0.6,
                   help="Weight given to the BM25 page-level score (λ). "
                        "GNN semantic weight = (1-λ).")
    p.add_argument("--top_k_bm25", type=int, default=10,
                   help="Number of BM25-retrieved candidate nodes passed to GNN reranker.")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit evaluation to first N questions (smoke test).")
    p.add_argument("--no_gnn", action="store_true",
                   help="Disable GNN reranking (BM25-only stage 1 oracle).")
    p.add_argument("--gnn_checkpoint", type=str,
                   default="checkpoints/hdgt_gnn_best.pt",
                   help="Path to trained GNN checkpoint.")
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 65)
    logger.info("  HDGT Phase 3 — Two-Stage Hybrid Retrieval Evaluation")
    logger.info("=" * 65)
    logger.info(f"  λ_bm25          = {args.lambda_bm25}")
    logger.info(f"  BM25 top-K      = {args.top_k_bm25}")
    logger.info(f"  GNN reranking   = {'OFF (BM25 only)' if args.no_gnn else 'ON'}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"  Device          = {device}")
    logger.info("=" * 65)

    # ------------------------------------------------------------------ #
    # 1.  Load models                                                       #
    # ------------------------------------------------------------------ #
    embedder = None
    gnn = None
    if not args.no_gnn:
        logger.info("Loading Qwen2.5-VL Question Embedder...")
        embedder = QwenVLEmbedder(device=str(device))

        logger.info("Loading HDGT Heterogeneous GNN...")
        gnn = HDGTHeteroGNN(in_dim=2048, hidden_dim=256, out_dim=128).to(device)
        ckpt = Path(args.gnn_checkpoint)
        if ckpt.exists():
            gnn.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            logger.info(f"  Loaded checkpoint: {ckpt}")
        else:
            logger.warning(f"  No checkpoint at {ckpt}, using random GNN weights.")
        gnn.eval()

    # ------------------------------------------------------------------ #
    # 2.  Data loaders                                                      #
    # ------------------------------------------------------------------ #
    qa_loader = MPDocVQALoader(data_root="data/MP-DocVQA", split="val")
    graph_loader = GraphLoader(graphs_dir="experiments/mpdocvqa")
    bm25_retriever = BM25Retriever()
    logger.info(f"Loaded {len(qa_loader)} validation questions.")

    # ------------------------------------------------------------------ #
    # 3.  Output file                                                        #
    # ------------------------------------------------------------------ #
    tag = f"hybrid_lam{args.lambda_bm25:.2f}_k{args.top_k_bm25}"
    if args.no_gnn:
        tag = "bm25_stage1_only"
    out_path = Path(f"experiments/retrieval_results_val_{tag}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 4.  Evaluation loop                                                    #
    # ------------------------------------------------------------------ #
    evaluated, skipped = 0, 0
    acc: dict = {k: 0.0 for k in ("recall@1", "recall@5", "recall@10", "mrr", "ela")}
    start_time = time.time()

    limit = args.limit or len(qa_loader)

    with open(out_path, "w") as fout:
        for idx, item in enumerate(tqdm(qa_loader, desc="Hybrid Retrieval", total=limit)):
            if idx >= limit or evaluated >= 3472:
                break

            if idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            question = item["question"]
            page_ids = item["page_ids"]
            gt_page = item.get("answer_page_idx", 0)
            gt_answers = item.get("answers", [])
            context_id = qa_loader.build_context_id(page_ids)

            g = graph_loader.load(context_id)
            if g is None:
                skipped += 1
                continue
            main._missing_count = 0

            # ----------------------------------------------------------
            # STAGE 1: BM25 — retrieve top-K candidate nodes
            # ----------------------------------------------------------
            try:
                bm25_candidates = bm25_retriever.retrieve(
                    question=question,
                    graph=g,
                    top_k=args.top_k_bm25,
                )
            except Exception as e:
                logger.warning(f"BM25 failed q_id={item.get('question_id', idx)}: {e}")
                skipped += 1
                continue

            if not bm25_candidates:
                skipped += 1
                continue

            if args.no_gnn:
                # BM25-only: return Stage 1 candidates as final ranking
                final_nodes = bm25_candidates[:10]
            else:
                # ----------------------------------------------------------
                # STAGE 2: GNN reranker — fuse BM25 + Qwen2.5-VL + graph
                # ----------------------------------------------------------
                has_qwen = (
                    "text" in g.node_types
                    and hasattr(g["text"], "qwen_x")
                    and g["text"].qwen_x is not None
                )
                if not has_qwen:
                    final_nodes = bm25_candidates[:10]
                else:
                    try:
                        # Query embedding
                        with torch.no_grad():
                            q_raw = embedder.embed_query(question).to(device)   # [1, 2048]
                            q_emb_proj = gnn.input_projections["text"](q_raw)
                            q_out = F.normalize(gnn.out_proj(q_emb_proj), p=2, dim=-1)  # [1, 128]

                            # Node features & edges
                            x_dict = {"text": g["text"].qwen_x.to(device)}
                            for nt in g.node_types:
                                if nt != "text" and hasattr(g[nt], "num_nodes") and g[nt].num_nodes > 0:
                                    x_dict[nt] = torch.zeros(
                                        (g[nt].num_nodes, 2048), device=device, dtype=torch.float32)

                            edge_index_dict = {}
                            for et in g.edge_types:
                                if hasattr(g[et], "edge_index") and g[et].edge_index.numel() > 0:
                                    edge_index_dict[et] = g[et].edge_index.to(device)

                            node_embs = gnn(x_dict, edge_index_dict)
                            text_embs = F.normalize(node_embs["text"], p=2, dim=-1)  # [N, 128]
                            gnn_scores = torch.sum(q_out * text_embs, dim=-1).cpu().tolist()

                        # Build BM25 page-level score index
                        bm25_page_scores: dict[int, float] = {}
                        for cand in bm25_candidates:
                            p = cand["page"]
                            if cand["score"] > bm25_page_scores.get(p, float("-inf")):
                                bm25_page_scores[p] = float(cand["score"])

                        max_bm25 = max(bm25_page_scores.values()) if bm25_page_scores else 1.0

                        # Fuse BM25 page score + GNN semantic score for ALL text nodes
                        n_text = g["text"].num_nodes
                        _mapping = g["text"]._mapping
                        pages_list = _mapping.get("pages", [0] * n_text)
                        contents_list = _mapping.get("contents", [""] * n_text)

                        final_candidates = []
                        for n_idx in range(n_text):
                            page = pages_list[n_idx] if n_idx < len(pages_list) else 0
                            content_str = contents_list[n_idx] if n_idx < len(contents_list) else ""

                            bm25_page_score = bm25_page_scores.get(page, 0.0) / (max_bm25 + 1e-8)
                            gnn_semantic_score = float(gnn_scores[n_idx]) if n_idx < len(gnn_scores) else 0.0

                            # λ * BM25_page + (1-λ) * GNN_semantic
                            hybrid_score = (args.lambda_bm25 * bm25_page_score
                                            + (1 - args.lambda_bm25) * gnn_semantic_score)

                            final_candidates.append({
                                "node_uid": f"{context_id}_n{n_idx}",
                                "type": "text",
                                "page": page,
                                "score": hybrid_score,
                                "bm25_page_score": bm25_page_score,
                                "gnn_score": gnn_semantic_score,
                                "content": content_str,
                            })

                        final_candidates.sort(key=lambda x: x["score"], reverse=True)
                        final_nodes = final_candidates[:10]

                        # Attach rank
                        for r, node in enumerate(final_nodes):
                            node["rank"] = r + 1

                    except Exception as e:
                        logger.warning(f"GNN rerank failed q_id={item.get('question_id', idx)}: {e}")
                        final_nodes = bm25_candidates[:10]

            # ----------------------------------------------------------
            # Metrics
            # ----------------------------------------------------------
            ret_pages = [n["page"] for n in final_nodes]
            ret_contents = [n.get("content", "") for n in final_nodes]
            try:
                item_metrics = compute_metrics(
                    retrieved_pages=ret_pages,
                    retrieved_contents=ret_contents,
                    ground_truth_page=gt_page,
                    ground_truth_answers=gt_answers,
                )
            except Exception as e:
                logger.warning(f"Metrics failed: {e}")
                skipped += 1
                continue

            for k in acc:
                acc[k] += item_metrics.get(k, 0.0)
            evaluated += 1

            fout.write(json.dumps({
                "question_id": item.get("question_id", str(idx)),
                "query": question,
                "context_id": context_id,
                "ground_truth_page": gt_page,
                "ground_truth_answers": gt_answers,
                "retrieved_nodes": final_nodes[:5],
                "metrics": item_metrics,
            }) + "\n")

    # ------------------------------------------------------------------ #
    # 5.  Print results                                                     #
    # ------------------------------------------------------------------ #
    elapsed = time.time() - start_time
    logger.info("=" * 65)
    logger.info("  HYBRID RETRIEVAL RESULTS")
    logger.info("=" * 65)
    logger.info(f"  Method          : {'BM25 (Stage 1 only)' if args.no_gnn else f'BM25 (λ={args.lambda_bm25:.2f}) + HDGT GNN'}")
    logger.info(f"  Evaluated       : {evaluated} / {min(limit, len(qa_loader))} questions")
    logger.info(f"  Skipped         : {skipped}")
    logger.info(f"  Time            : {elapsed:.1f}s ({elapsed/max(evaluated, 1):.3f}s/q)")
    logger.info("-" * 65)
    if evaluated:
        for k in ("recall@1", "recall@5", "recall@10", "mrr", "ela"):
            v = acc[k] / evaluated
            logger.info(f"  {k.upper():<14} : {v:.4f}  ({v*100:.2f}%)")
    logger.info("=" * 65)
    logger.info(f"  Results saved to: {out_path}")
    logger.info("=" * 65)

    # Print comparison table
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│   PHASE COMPARISON TABLE (MP-DocVQA Validation Set)          │")
    print("├────────────────┬───────────────┬──────────────┬──────────────┤")
    print("│ Metric         │ BM25 Baseline │ Qwen+GNN P2  │ Hybrid P3    │")
    print("├────────────────┼───────────────┼──────────────┼──────────────┤")
    if evaluated:
        print(f"│ Recall@1       │    59.73%     │    51.08%    │  {acc['recall@1']/evaluated*100:6.2f}%    │")
        print(f"│ Recall@5       │    76.96%     │    51.08%    │  {acc['recall@5']/evaluated*100:6.2f}%    │")
        print(f"│ Recall@10      │    82.36%     │    51.08%    │  {acc['recall@10']/evaluated*100:6.2f}%    │")
        print(f"│ MRR            │    0.6701     │    0.5108    │   {acc['mrr']/evaluated:.4f}    │")
        print(f"│ ELA            │    0.0052     │    0.0103    │   {acc['ela']/evaluated:.4f}    │")
    print("└────────────────┴───────────────┴──────────────┴──────────────┘\n")


if __name__ == "__main__":
    main()
