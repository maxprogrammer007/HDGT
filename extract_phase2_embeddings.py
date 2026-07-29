"""
extract_phase2_embeddings.py

Phase 2 — Batch node feature extraction using Qwen2.5-VL over HDGT graphs.
"""

import glob
import logging
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from hdgt.models import QwenVLEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("extract_phase2_embeddings")


def main():
    data_root = Path("data/MP-DocVQA")
    graphs_dir = Path("experiments/mpdocvqa")
    imdb_file = data_root / "mpdocvqa_imdbs" / "imdb_val.npy"

    if not imdb_file.exists():
        logger.error(f"IMDB file not found at {imdb_file}")
        sys.exit(1)

    logger.info(f"Loading IMDB OCR dataset: {imdb_file.name}")
    imdb_data = np.load(imdb_file, allow_pickle=True)[1:]

    # Map context_id -> list of all text tokens across pages
    context_tokens_map = {}
    for item in imdb_data:
        doc_id = item.get("image_id", "")
        pages = item.get("pages", [])
        ocr_tokens = item.get("ocr_tokens", [])
        if not pages or not ocr_tokens:
            continue
        p_min = min(pages)
        p_max = max(pages)
        context_id = f"{doc_id}_p{p_min}_p{p_max}"

        # Flatten tokens across pages
        all_tokens = []
        for pg_tokens in ocr_tokens:
            all_tokens.extend([str(t).strip() for t in pg_tokens if str(t).strip()])
        
        context_tokens_map[context_id] = all_tokens

    graph_paths = sorted(list(graphs_dir.glob("*_graph.pt")))
    logger.info(f"Found {len(graph_paths)} graph files for feature extraction.")

    logger.info("Initializing Qwen2.5-VL Embedder on available GPUs...")
    embedder = QwenVLEmbedder(model_id="Qwen/Qwen2.5-VL-3B-Instruct")
    logger.info(f"Embedder initialized. Hidden dimension: {embedder.hidden_dim}")

    processed_count = 0
    skipped_count = 0
    fail_count = 0

    for path in tqdm(graph_paths, desc="Extracting Qwen2.5-VL Embeddings"):
        try:
            g = torch.load(path, weights_only=False)
            
            # Check if text nodes feature 'qwen_x' already exists
            if hasattr(g["text"], "qwen_x") and g["text"].qwen_x is not None:
                skipped_count += 1
                continue

            stem = path.stem.replace("_graph", "")
            tokens = context_tokens_map.get(stem, [])

            num_text_nodes = g["text"].num_nodes if hasattr(g["text"], "num_nodes") else 0
            if num_text_nodes == 0:
                skipped_count += 1
                continue

            # Ensure token count matches node count or handle truncation/padding
            if len(tokens) < num_text_nodes:
                tokens = tokens + [" "] * (num_text_nodes - len(tokens))
            elif len(tokens) > num_text_nodes:
                tokens = tokens[:num_text_nodes]

            qwen_x = embedder.embed_texts(tokens, batch_size=64)
            g["text"].qwen_x = qwen_x

            # Save updated graph
            torch.save(g, path)
            processed_count += 1

        except Exception as exc:
            logger.warning(f"Failed feature extraction for {path.name}: {exc}")
            fail_count += 1

    logger.info("=" * 60)
    logger.info("  Phase 2 Feature Extraction Complete")
    logger.info(f"  Processed          : {processed_count}")
    logger.info(f"  Already had features: {skipped_count}")
    logger.info(f"  Failed             : {fail_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
