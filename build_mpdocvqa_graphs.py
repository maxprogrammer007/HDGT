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
import sys
from pathlib import Path

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
    print(f"  Data root  : {root_dir}")
    print(f"  Output dir : {output_dir}")
    print(f"  Split      : {args.split}")
    print(f"  Limit      : {args.limit or 'All'}")
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

    # ── Parser (shared across all documents) ─────────────────────────────
    # save_figures=False: page images already exist in images/
    # no need to re-extract figure crops for the graph builder.
    parser_obj = DoclingParser(verbose=args.verbose, save_figures=False)

    success_count = 0
    fail_count    = 0
    skip_count    = 0

    for context_id, ctx_info in tqdm(contexts.items(), desc="Building graphs"):
        pdf_path   = root_dir / "pdfs" / f"{context_id}.pdf"
        graph_path = output_dir / f"{context_id}_graph.pt"

        # Skip already-built graphs
        if graph_path.exists():
            skip_count += 1
            continue

        # Skip if PDF not compiled yet (images unavailable)
        if not pdf_path.exists():
            logger.debug(f"PDF not found: {pdf_path}")
            fail_count += 1
            continue

        try:
            # Step 1: Parse PDF
            nodes = parser_obj.parse(pdf_path, document_id=context_id)

            # Step 2: Build node features
            node_builder = NodeBuilder()
            node_builder.build(nodes)

            # Step 3: Build edges
            edge_builder = EdgeBuilder(edge_cfg)
            edges        = edge_builder.build(nodes, node_builder)

            # Step 4: Assemble HeteroData
            data = build_hetero_data(node_builder, edges)
            data.validate()

            # Step 5: Save
            torch.save(data, graph_path)
            success_count += 1

        except Exception as e:
            logger.warning(f"Failed for {context_id}: {e}")
            fail_count += 1

        # Periodic GC to prevent memory bloat on large batches
        if success_count > 0 and success_count % args.gc_every == 0:
            gc.collect()

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  GRAPH CONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"  Successfully built : {success_count:,}")
    print(f"  Already existed    : {skip_count:,}")
    print(f"  Failed / missing   : {fail_count:,}")
    print(f"  Output directory   : {output_dir}")
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
