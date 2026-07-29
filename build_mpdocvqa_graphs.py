"""
build_mpdocvqa_graphs.py

Phase 1.6 — Build PyG HeteroData graphs for all MP-DocVQA contexts.

Reads the compiled multi-page PDFs from {data_root}/pdfs/, parses each
through DoclingParser, and saves a HeteroData .pt file per context.

Usage
-----
# On the workstation (after prepare_mpdocvqa.py):
python build_mpdocvqa_graphs.py \\
    --data-root /home/cvpruts/Downloads/HDGT-main\\(1\\)/HDGT-main/data/MP-DocVQA \\
    --output-dir experiments/mpdocvqa

# Limit to first 100 contexts (smoke test):
python build_mpdocvqa_graphs.py \\
    --data-root /home/cvpruts/Downloads/HDGT-main\\(1\\)/HDGT-main/data/MP-DocVQA \\
    --limit 100

# Filter to val split only:
python build_mpdocvqa_graphs.py \\
    --data-root /home/cvpruts/Downloads/HDGT-main\\(1\\)/HDGT-main/data/MP-DocVQA \\
    --split val
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Force CPU mode for Docling/RapidOCR worker processes to prevent CUDA fork errors in multiprocessing
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from hdgt.parsers.docling_parser import DoclingParser
from hdgt.graph.node_builder import NodeBuilder
from hdgt.graph.edge_builder import EdgeBuilder
from hdgt.graph.hetero_graph import build_hetero_data

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("build_mpdocvqa_graphs")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HDGT — MP-DocVQA Graph Construction Loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        default="data/MP-DocVQA",
        help=(
            "Path to the MP-DocVQA data directory (must contain context_map.json "
            "and pdfs/ subdirectory). "
            "Workstation path: "
            "/home/cvpruts/Downloads/HDGT-main(1)/HDGT-main/data/MP-DocVQA"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/mpdocvqa",
        help="Directory to save *_graph.pt files.",
    )
    parser.add_argument(
        "--config",
        default="hdgt/configs/default.yaml",
        help="Path to the HDGT config YAML.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Only build graphs for contexts from a specific split.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Build at most N graphs. Useful for smoke testing.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel worker processes for graph construction.",
    )
    parser.add_argument(
        "--gc-every",
        type=int,
        default=50,
        help="Run garbage collection every N successfully built graphs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose Docling parser output.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Worker function for parallel processing
# ---------------------------------------------------------------------------

_worker_parser: Optional[DoclingParser] = None

def _init_worker(verbose: bool) -> None:
    global _worker_parser
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        _worker_parser = DoclingParser(verbose=verbose, save_figures=False)
    except Exception as exc:
        logger.warning(f"Worker parser init warning: {exc}")

def _build_single_graph(item: tuple) -> dict:
    global _worker_parser
    if _worker_parser is None:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        _worker_parser = DoclingParser(verbose=False, save_figures=False)

    context_id, pdf_path_str, graph_path_str, edge_cfg, verbose = item
    pdf_path   = Path(pdf_path_str)
    graph_path = Path(graph_path_str)

    if graph_path.exists():
        try:
            data = torch.load(graph_path, weights_only=False)
            n_nodes = sum(data[nt].num_nodes for nt in data.node_types if hasattr(data[nt], "num_nodes"))
            n_edges = sum(data[et].num_edges for et in data.edge_types if hasattr(data[et], "num_edges"))
            return {"status": "skip", "nodes": n_nodes, "edges": n_edges, "time": 0.0}
        except Exception:
            return {"status": "skip", "nodes": 0, "edges": 0, "time": 0.0}

    if not pdf_path.exists():
        return {"status": "missing", "nodes": 0, "edges": 0, "time": 0.0}

    start_t = time.time()
    try:
        nodes = _worker_parser.parse(pdf_path, document_id=context_id)

        node_builder = NodeBuilder()
        node_builder.build(nodes)

        edge_builder = EdgeBuilder(edge_cfg)
        edges        = edge_builder.build(nodes, node_builder)

        data = build_hetero_data(node_builder, edges)
        data.validate()

        torch.save(data, graph_path)
        elapsed = time.time() - start_t
        n_nodes = sum(data[nt].num_nodes for nt in data.node_types if hasattr(data[nt], "num_nodes"))
        n_edges = sum(data[et].num_edges for et in data.edge_types if hasattr(data[et], "num_edges"))
        return {"status": "success", "nodes": n_nodes, "edges": n_edges, "time": elapsed}
    except Exception as e:
        logger.warning(f"Failed for {context_id}: {e}")
        return {"status": "fail", "nodes": 0, "edges": 0, "time": 0.0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    root_dir   = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  HDGT — MP-DocVQA Graph Construction")
    print("=" * 60)
    print(f"  Data root   : {root_dir}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Split       : {args.split}")
    print(f"  Limit       : {args.limit or 'All'}")
    print(f"  Num workers : {args.num_workers}")
    print("=" * 60)

    # ── Load config ─────────────────────────────────────────────────────
    config_path = Path(args.config)
    if config_path.exists():
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    else:
        logger.warning(f"Config not found at {config_path}, using defaults.")
        config = {}

    edge_cfg = config.get("edges", {})

    # ── Load context map ─────────────────────────────────────────────────
    mapping_file = root_dir / "context_map.json"
    if not mapping_file.exists():
        print(f"\n[ERROR] context_map.json not found at {mapping_file}")
        print("  Run: python prepare_mpdocvqa.py --data-root", args.data_root)
        sys.exit(1)

    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    contexts = mapping_data["contexts"]

    # ── Filter by split ───────────────────────────────────────────────────
    if args.split != "all":
        contexts = {
            cid: ctx
            for cid, ctx in contexts.items()
            if ctx.get("split", "") == args.split
        }

    # ── Apply limit ───────────────────────────────────────────────────────
    if args.limit is not None:
        contexts = dict(list(contexts.items())[: args.limit])

    print(f"\n  Processing {len(contexts):,} contexts...")

    work_items = [
        (
            context_id,
            str(root_dir / "pdfs" / f"{context_id}.pdf"),
            str(output_dir / f"{context_id}_graph.pt"),
            edge_cfg,
            args.verbose,
        )
        for context_id in contexts.keys()
    ]

    success_count = 0
    fail_count    = 0
    skip_count    = 0
    total_nodes   = 0
    total_edges   = 0
    total_time    = 0.0

    def process_res(res: dict):
        nonlocal success_count, fail_count, skip_count, total_nodes, total_edges, total_time
        st = res.get("status", "fail")
        if st == "success":
            success_count += 1
            total_nodes += res.get("nodes", 0)
            total_edges += res.get("edges", 0)
            total_time  += res.get("time", 0.0)
        elif st == "skip":
            skip_count += 1
            total_nodes += res.get("nodes", 0)
            total_edges += res.get("edges", 0)
        else:
            fail_count += 1

    if args.num_workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(args.verbose,),
        ) as executor:
            futures = [executor.submit(_build_single_graph, item) for item in work_items]
            for future in tqdm(as_completed(futures), total=len(work_items), desc="Building graphs (parallel)"):
                try:
                    process_res(future.result())
                except Exception as exc:
                    logger.warning(f"Worker process error: {exc}")
                    fail_count += 1
    else:
        for item in tqdm(work_items, desc="Building graphs (sequential)"):
            try:
                process_res(_build_single_graph(item))
            except Exception as exc:
                logger.warning(f"Build error: {exc}")
                fail_count += 1

    total_processed = success_count + skip_count
    avg_nodes = (total_nodes / total_processed) if total_processed > 0 else 0.0
    avg_edges = (total_edges / total_processed) if total_processed > 0 else 0.0
    avg_time  = (total_time / success_count) if success_count > 0 else 0.0

    stats_dict = {
        "split": args.split,
        "total_contexts": len(contexts),
        "successfully_built": success_count,
        "already_existed": skip_count,
        "failed_or_missing": fail_count,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "avg_nodes_per_graph": round(avg_nodes, 2),
        "avg_edges_per_graph": round(avg_edges, 2),
        "avg_build_time_sec": round(avg_time, 3),
        "success_rate": round(100.0 * (success_count + skip_count) / max(1, len(contexts)), 2)
    }

    stats_file = Path("phase1_6_results") / f"graph_construction_stats_{args.split}.json"
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  GRAPH CONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"  Successfully built : {success_count:,}")
    print(f"  Already existed    : {skip_count:,}")
    print(f"  Failed / missing   : {fail_count:,}")
    print(f"  Avg nodes / graph  : {avg_nodes:.1f}")
    print(f"  Avg edges / graph  : {avg_edges:.1f}")
    if success_count > 0:
        print(f"  Avg build time     : {avg_time:.3f} sec/graph")
    print(f"  Output directory   : {output_dir}")
    print(f"  Stats saved to     : {stats_file}")
    print("=" * 60)

    if success_count + skip_count == 0 and fail_count > 0:
        print(
            "\n[WARNING] All contexts failed.\n"
            "  Make sure PDFs were compiled:\n"
            f"  python prepare_mpdocvqa.py --data-root {args.data_root}"
        )

    print(
        "\nNext step:\n"
        f"  python evaluate_retrieval.py "
        f"--data-root {args.data_root} "
        f"--graphs-dir {args.output_dir} "
        f"--method bm25 --split val"
    )


if __name__ == "__main__":
    main()
