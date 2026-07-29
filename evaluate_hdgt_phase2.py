"""
evaluate_hdgt_phase2.py

HDGT Phase 2 — Multimodal Qwen2.5-VL + Heterogeneous GNN Evaluation Benchmark.
"""

import json
import logging
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm

from hdgt.models import QwenVLEmbedder, HDGTHeteroGNN
from hdgt.evaluation.loaders import MPDocVQALoader, GraphLoader
from hdgt.evaluation.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("evaluate_hdgt_phase2")


def main():
    logger.info("=" * 60)
    logger.info("  HDGT Phase 2 — Qwen2.5-VL + GNN Multimodal Retrieval Benchmark")
    logger.info("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Using compute device: {device}")

    # Load Qwen2.5-VL embedder for question queries
    logger.info("Loading Qwen2.5-VL Question Embedder...")
    embedder = QwenVLEmbedder(device=str(device))

    # Load HDGT Heterogeneous GNN model
    logger.info("Initializing HDGT Heterogeneous GNN Model...")
    gnn = HDGTHeteroGNN(in_dim=2048, hidden_dim=256, out_dim=128).to(device)
    
    ckpt_path = Path("checkpoints/hdgt_gnn_best.pt")
    if ckpt_path.exists():
        logger.info(f"Loading trained checkpoint from {ckpt_path}...")
        gnn.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    else:
        logger.warning(f"No checkpoint found at {ckpt_path}, using initial weights.")

    gnn.eval()

    qa_loader = MPDocVQALoader(data_root="data/MP-DocVQA", split="val")
    graph_loader = GraphLoader(graphs_dir="experiments/mpdocvqa")
    logger.info(f"Loaded {len(qa_loader)} validation questions from MP-DocVQA.")

    results_file = Path("experiments/retrieval_results_val_phase2_hdgt.jsonl")
    results_file.parent.mkdir(parents=True, exist_ok=True)

    evaluated = 0
    skipped = 0
    metrics_accumulator = {
        "recall@1": 0.0,
        "recall@5": 0.0,
        "recall@10": 0.0,
        "mrr": 0.0,
        "ela": 0.0,
        "anls": 0.0,
    }

    start_time = time.time()

    with open(results_file, "w", encoding="utf-8") as f_out:
        for idx, item in enumerate(tqdm(qa_loader, desc="Phase 2 Evaluation")):
            question_id = item.get("question_id", str(idx))
            query = item["question"]
            page_ids = item["page_ids"]
            gt_answers = item.get("answers", [])
            gt_page_idx = item.get("answer_page_idx", 0)

            context_id = qa_loader.build_context_id(page_ids)
            g = graph_loader.load(context_id)

            if g is None or "text" not in g.node_types or not hasattr(g["text"], "qwen_x") or g["text"].qwen_x is None:
                skipped += 1
                continue

            num_text_nodes = g["text"].num_nodes if hasattr(g["text"], "num_nodes") else 0
            if num_text_nodes == 0:
                skipped += 1
                continue

            try:
                # 1. Embed question query with Qwen2.5-VL
                q_emb = embedder.embed_query(query).to(device)  # [1, 2048]
                q_emb_proj = gnn.input_projections["text"](q_emb)  # [1, 256]
                q_out = F.normalize(gnn.out_proj(q_emb_proj), p=2, dim=-1)  # [1, 128]

                # 2. Extract graph node features and run GNN message passing
                x_dict = {"text": g["text"].qwen_x.to(device)}
                
                # Add default features for other node types if needed
                for nt in g.node_types:
                    if nt != "text" and hasattr(g[nt], "num_nodes") and g[nt].num_nodes > 0:
                        x_dict[nt] = torch.zeros((g[nt].num_nodes, 2048), device=device, dtype=torch.float32)

                edge_index_dict = {}
                for et in g.edge_types:
                    if hasattr(g[et], "edge_index") and g[et].edge_index.numel() > 0:
                        edge_index_dict[et] = g[et].edge_index.to(device)

                # Forward pass through HDGT HeteroGNN
                with torch.no_grad():
                    node_embs_dict = gnn(x_dict, edge_index_dict)
                    text_embs = F.normalize(node_embs_dict["text"], p=2, dim=-1)  # [N_text, 128]

                    # 3. Compute cosine relevance scores between query and graph text nodes
                    scores = torch.sum(q_out * text_embs, dim=-1).cpu().numpy()

                # Rank nodes by relevance score
                top_indices = torch.argsort(torch.tensor(scores), descending=True).tolist()
                
                retrieved_nodes = []
                # Use _mapping to correctly access list-valued attributes
                _mapping = g["text"]._mapping
                pages_list    = _mapping.get("pages",    [0] * num_text_nodes)
                contents_list = _mapping.get("contents", [""] * num_text_nodes)

                for rank_idx, node_i in enumerate(top_indices[:10]):
                    page_num    = pages_list[node_i]    if node_i < len(pages_list)    else 0
                    content_str = contents_list[node_i] if node_i < len(contents_list) else ""

                    retrieved_nodes.append({
                        "node_uid": f"{context_id}_n{node_i}",
                        "type": "text",
                        "page": page_num,
                        "rank": rank_idx + 1,
                        "score": float(scores[node_i]),
                        "content": content_str,
                    })

                # Compute evaluation metrics
                retrieved_pages = [n["page"] for n in retrieved_nodes]
                retrieved_contents = [n["content"] for n in retrieved_nodes]
                item_metrics = compute_metrics(
                    retrieved_pages=retrieved_pages,
                    retrieved_contents=retrieved_contents,
                    ground_truth_page=gt_page_idx,
                    ground_truth_answers=gt_answers,
                )

                for k, v in item_metrics.items():
                    metrics_accumulator[k] += v

                evaluated += 1

                record = {
                    "question_id": question_id,
                    "query": query,
                    "context_id": context_id,
                    "ground_truth_page": gt_page_idx,
                    "ground_truth_answers": gt_answers,
                    "retrieved_nodes": retrieved_nodes[:5],
                    "metrics": item_metrics,
                }
                f_out.write(json.dumps(record) + "\n")

            except Exception as exc:
                logger.warning(f"Error evaluating q_id={question_id}: {exc}")
                skipped += 1

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("  PHASE 2 RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Method         : Qwen2.5-VL + HDGT GNN")
    logger.info(f"  Split          : val")
    logger.info(f"  Evaluated      : {evaluated} / {len(qa_loader)} questions")
    logger.info(f"  Skipped        : {skipped}")
    logger.info(f"  Elapsed Time   : {elapsed:.2f}s ({elapsed/max(1, evaluated):.3f}s / question)")
    logger.info("-" * 60)

    if evaluated > 0:
        for k in metrics_accumulator:
            avg_v = metrics_accumulator[k] / evaluated
            logger.info(f"  {k.upper():<14} : {avg_v:.4f}")
    logger.info("=" * 60)
    logger.info(f"  Results saved  : {results_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
