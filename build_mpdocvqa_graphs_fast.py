"""
build_mpdocvqa_graphs_fast.py

Fast layout graph builder using MP-DocVQA pre-extracted Amazon Textract OCR
data from mpdocvqa_imdbs/imdb_{split}.npy.

Constructs PyG HeteroData graphs for all contexts in seconds.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from hdgt.graph.schema import DocumentNode
from hdgt.graph.node_builder import NodeBuilder
from hdgt.graph.edge_builder import EdgeBuilder
from hdgt.graph.hetero_graph import build_hetero_data

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("build_mpdocvqa_graphs_fast")


def parse_args():
    p = argparse.ArgumentParser(description="Fast MP-DocVQA Graph Construction")
    p.add_argument("--data-root", default="data/MP-DocVQA")
    p.add_argument("--output-dir", default="experiments/mpdocvqa")
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    p.add_argument("--config", default="hdgt/configs/default.yaml")
    return p.parse_args()


def build_graphs(args):
    root_dir   = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  HDGT — Fast MP-DocVQA Graph Builder (IMDB OCR Data)")
    print("=" * 60)

    # Load edge config
    import yaml
    config_path = Path(args.config)
    edge_cfg = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            edge_cfg = (yaml.safe_load(f) or {}).get("edges", {})

    imdb_file = root_dir / "mpdocvqa_imdbs" / f"imdb_{args.split}.npy"
    if not imdb_file.exists():
        print(f"[ERROR] IMDB file not found at {imdb_file}")
        sys.exit(1)

    print(f"  Loading IMDB split: {imdb_file.name}")
    imdb_data = np.load(imdb_file, allow_pickle=True)

    # Group QA items by context_id
    contexts_items = {}
    for item in imdb_data:
        doc_id = item.get("image_id", "")
        pages = item.get("pages", [])
        if not pages:
            continue
        p_min = min(pages)
        p_max = max(pages)
        context_id = f"{doc_id}_p{p_min}_p{p_max}"
        if context_id not in contexts_items:
            contexts_items[context_id] = item

    print(f"  Found {len(contexts_items):,} unique contexts in split '{args.split}'")

    success_count = 0
    skip_count    = 0
    fail_count    = 0
    total_nodes   = 0
    total_edges   = 0
    start_time    = time.time()

    work_items = list(contexts_items.items())

    def _build_single_item(item_tuple):
        context_id, item = item_tuple
        graph_path = output_dir / f"{context_id}_graph.pt"
        if graph_path.exists():
            try:
                g = torch.load(graph_path, weights_only=False)
                n_nodes = sum(g[nt].num_nodes for nt in g.node_types if hasattr(g[nt], "num_nodes"))
                n_edges = sum(g[et].num_edges for et in g.edge_types if hasattr(g[et], "num_edges"))
                return ("skip", n_nodes, n_edges)
            except Exception:
                pass

        try:
            tokens_per_page = item.get("ocr_tokens", [])
            boxes_per_page  = item.get("ocr_normalized_boxes", [])
            doc_id          = item.get("image_id", "")

            nodes = []
            node_id_counter = 0

            for pg_idx, (tokens, boxes) in enumerate(zip(tokens_per_page, boxes_per_page)):
                # Page node
                page_uid = f"{doc_id}_p{pg_idx}_n{node_id_counter}"
                nodes.append(DocumentNode(
                    node_id=node_id_counter,
                    node_uid=page_uid,
                    document_id=doc_id,
                    page=pg_idx,
                    type="page",
                    role="page",
                    bbox=[0.0, 0.0, 1.0, 1.0],
                    content=f"Page {pg_idx + 1}",
                ))
                node_id_counter += 1

                # Text nodes
                for tok, box in zip(tokens, boxes):
                    tok_str = str(tok).strip()
                    if not tok_str:
                        continue
                    b = [max(0.0, min(1.0, float(c))) for c in box]
                    uid = f"{doc_id}_p{pg_idx}_n{node_id_counter}"
                    nodes.append(DocumentNode(
                        node_id=node_id_counter,
                        node_uid=uid,
                        document_id=doc_id,
                        page=pg_idx,
                        type="text",
                        role="paragraph",
                        bbox=b,
                        content=tok_str,
                    ))
                    node_id_counter += 1

            node_builder = NodeBuilder()
            node_builder.build(nodes)

            edge_builder = EdgeBuilder(edge_cfg)
            edges        = edge_builder.build(nodes, node_builder)

            data = build_hetero_data(node_builder, edges)
            data.validate()

            torch.save(data, graph_path)
            n_nodes = sum(data[nt].num_nodes for nt in data.node_types if hasattr(data[nt], "num_nodes"))
            n_edges = sum(data[et].num_edges for et in data.edge_types if hasattr(data[et], "num_edges"))
            return ("success", n_nodes, n_edges)
        except Exception as exc:
            logger.warning(f"Failed for {context_id}: {exc}")
            return ("fail", 0, 0)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_build_single_item, item) for item in work_items]
        for future in tqdm(as_completed(futures), total=len(work_items), desc="Building graphs"):
            status, n_nodes, n_edges = future.result()
            if status == "success":
                success_count += 1
                total_nodes += n_nodes
                total_edges += n_edges
            elif status == "skip":
                skip_count += 1
                total_nodes += n_nodes
                total_edges += n_edges
            else:
                fail_count += 1

    elapsed = time.time() - start_time
    total_processed = success_count + skip_count
    avg_nodes = (total_nodes / total_processed) if total_processed > 0 else 0.0
    avg_edges = (total_edges / total_processed) if total_processed > 0 else 0.0

    print("\n" + "=" * 60)
    print("  FAST GRAPH CONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"  Successfully built : {success_count:,}")
    print(f"  Already existed    : {skip_count:,}")
    print(f"  Failed / missing   : {fail_count:,}")
    print(f"  Avg nodes / graph  : {avg_nodes:.1f}")
    print(f"  Avg edges / graph  : {avg_edges:.1f}")
    print(f"  Total elapsed time : {elapsed:.2f} seconds")
    print(f"  Output directory   : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    build_graphs(parse_args())
