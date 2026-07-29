"""
train_hdgt_phase2.py

HDGT Phase 2 — Train Heterogeneous GNN for Multimodal Sub-Graph Retrieval.
"""

import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from hdgt.models import QwenVLEmbedder, HDGTHeteroGNN
from hdgt.evaluation.loaders import MPDocVQALoader, GraphLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("train_hdgt_phase2")


def contrastive_loss(q_emb: torch.Tensor, node_embs: torch.Tensor, gt_node_idx: int, temp: float = 0.07) -> torch.Tensor:
    """
    InfoNCE Contrastive Retrieval Loss.
    q_emb: [1, dim]
    node_embs: [N_nodes, dim]
    gt_node_idx: Index of ground-truth target text node
    """
    sims = torch.sum(q_emb * node_embs, dim=-1) / temp  # [N_nodes]
    target = torch.tensor([gt_node_idx], device=sims.device)
    loss = F.cross_entropy(sims.unsqueeze(0), target)
    return loss


def main():
    logger.info("=" * 60)
    logger.info("  HDGT Phase 2 — Multimodal GNN Contrastive Training")
    logger.info("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize Qwen Embedder (frozen for fast GNN training)
    logger.info("Loading Qwen2.5-VL Question Embedder...")
    embedder = QwenVLEmbedder(device=str(device))

    # Initialize HDGT HeteroGNN
    gnn = HDGTHeteroGNN(in_dim=2048, hidden_dim=256, out_dim=128).to(device)
    optimizer = AdamW(gnn.parameters(), lr=1e-3, weight_decay=1e-4)

    # Use validation set items for quick fine-tuning demonstration
    qa_loader = MPDocVQALoader(data_root="data/MP-DocVQA", split="val")
    graph_loader = GraphLoader(graphs_dir="experiments/mpdocvqa")
    logger.info(f"Loaded {len(qa_loader)} items for training.")

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "hdgt_gnn_best.pt"

    epochs = 3
    for epoch in range(1, epochs + 1):
        gnn.train()
        total_loss = 0.0
        step_count = 0
        start_time = time.time()

        for idx, item in enumerate(tqdm(qa_loader, desc=f"Epoch {epoch}/{epochs}")):
            query = item["question"]
            page_ids = item["page_ids"]
            gt_page_idx = item.get("answer_page_idx", 0)

            context_id = qa_loader.build_context_id(page_ids)
            g = graph_loader.load(context_id)

            if g is None or "text" not in g.node_types or not hasattr(g["text"], "qwen_x") or g["text"].qwen_x is None:
                continue

            num_text_nodes = g["text"].num_nodes if hasattr(g["text"], "num_nodes") else 0
            if num_text_nodes == 0:
                continue

            # Find target node matching ground truth page
            gt_node_indices = []
            if hasattr(g["text"], "page"):
                for n_idx, p in enumerate(g["text"].page):
                    if int(p.item()) == gt_page_idx:
                        gt_node_indices.append(n_idx)

            if not gt_node_indices:
                gt_node_idx = 0
            else:
                gt_node_idx = gt_node_indices[0]

            optimizer.zero_grad()

            # Question Embedding
            with torch.no_grad():
                q_raw = embedder.embed_query(query).to(device)  # [1, 2048]

            q_emb_proj = gnn.input_projections["text"](q_raw)
            q_out = F.normalize(gnn.out_proj(q_emb_proj), p=2, dim=-1)

            # Node Features & Edges
            x_dict = {"text": g["text"].qwen_x.to(device)}
            for nt in g.node_types:
                if nt != "text" and hasattr(g[nt], "num_nodes") and g[nt].num_nodes > 0:
                    x_dict[nt] = torch.zeros((g[nt].num_nodes, 2048), device=device, dtype=torch.float32)

            edge_index_dict = {}
            for et in g.edge_types:
                if hasattr(g[et], "edge_index") and g[et].edge_index.numel() > 0:
                    edge_index_dict[et] = g[et].edge_index.to(device)

            # Forward Pass
            node_embs_dict = gnn(x_dict, edge_index_dict)
            text_embs = F.normalize(node_embs_dict["text"], p=2, dim=-1)

            loss = contrastive_loss(q_out, text_embs, gt_node_idx)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step_count += 1

            if step_count >= 500:  # Train on 500 batches per epoch for speed
                break

        avg_loss = total_loss / max(1, step_count)
        elapsed = time.time() - start_time
        logger.info(f"Epoch {epoch}/{epochs} Complete | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

    # Save trained checkpoint
    torch.save(gnn.state_dict(), best_ckpt_path)
    logger.info(f"Successfully saved trained GNN checkpoint to {best_ckpt_path}")


if __name__ == "__main__":
    main()
