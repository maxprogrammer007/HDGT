"""
graph/hetero_graph.py

Assembles a torch_geometric.data.HeteroData object from NodeBuilder output
and a list of DocumentEdge objects.

Structure of the output HeteroData:
  data[node_type].x          — float32 feature tensor (N, 9)
  data[node_type].node_ids   — List[int]  global node_ids
  data[node_type].node_uids  — List[str]  globally unique string ids

  data[src_type, relation, dst_type].edge_index   — LongTensor (2, E)
  data[src_type, relation, dst_type].edge_weight  — FloatTensor (E,)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch_geometric.data import HeteroData

from hdgt.graph.schema import DocumentEdge, DocumentNode
from hdgt.graph.node_builder import NodeBuilder

logger = logging.getLogger(__name__)

# Type alias for edge key
EdgeKey = Tuple[str, str, str]   # (src_type, relation, dst_type)


def build_hetero_data(
    node_builder: NodeBuilder,
    edges: List[DocumentEdge],
) -> HeteroData:
    """
    Build a PyG HeteroData object from NodeBuilder results and DocumentEdge list.

    Parameters
    ----------
    node_builder : NodeBuilder
        Must have called node_builder.build() beforehand.
    edges : List[DocumentEdge]
        All edges produced by EdgeBuilder.build().

    Returns
    -------
    HeteroData
        Ready to pass to PyG GNN layers in Phase 3.
    """
    data = HeteroData()

    # ── Node features ─────────────────────────────────────────────────
    for ntype, feat_tensor in node_builder.features.items():
        data[ntype].x         = feat_tensor
        data[ntype].node_ids  = [n.node_id  for n in node_builder.node_lists[ntype]]
        data[ntype].node_uids = [n.node_uid for n in node_builder.node_lists[ntype]]
        data[ntype].roles     = [n.role     for n in node_builder.node_lists[ntype]]
        # Phase 1.6: raw text metadata for retrieval evaluation
        data[ntype].contents  = [n.content  for n in node_builder.node_lists[ntype]]
        data[ntype].bboxes    = [n.bbox     for n in node_builder.node_lists[ntype]]
        data[ntype].pages     = [n.page     for n in node_builder.node_lists[ntype]]

    # ── Edges ──────────────────────────────────────────────────────────
    # Accumulate per (src_type, relation, dst_type)
    edge_buckets: Dict[EdgeKey, Dict] = defaultdict(lambda: {
        "srcs": [], "dsts": [], "weights": []
    })

    skipped = 0
    for edge in edges:
        # Translate global IDs → per-type local indices
        if edge.src_id not in node_builder.global_to_local:
            skipped += 1
            continue
        if edge.dst_id not in node_builder.global_to_local:
            skipped += 1
            continue

        src_type, src_local = node_builder.global_to_local[edge.src_id]
        dst_type, dst_local = node_builder.global_to_local[edge.dst_id]
        key: EdgeKey = (src_type, edge.relation, dst_type)

        edge_buckets[key]["srcs"].append(src_local)
        edge_buckets[key]["dsts"].append(dst_local)
        edge_buckets[key]["weights"].append(edge.weight)

    if skipped:
        logger.warning(f"Skipped {skipped} edges with unknown node IDs.")

    for (src_type, relation, dst_type), bucket in edge_buckets.items():
        srcs = torch.tensor(bucket["srcs"],   dtype=torch.long)
        dsts = torch.tensor(bucket["dsts"],   dtype=torch.long)
        wts  = torch.tensor(bucket["weights"], dtype=torch.float32)

        data[src_type, relation, dst_type].edge_index  = torch.stack([srcs, dsts], dim=0)
        data[src_type, relation, dst_type].edge_weight = wts

    logger.info(
        f"HeteroData: {len(node_builder.features)} node types, "
        f"{len(edge_buckets)} edge types."
    )
    return data


def print_heterodata_summary(data: HeteroData) -> None:
    """Print a human-readable summary of a HeteroData object."""
    print("\n" + "=" * 60)
    print("  HETERODATA SUMMARY")
    print("=" * 60)

    print("\n  Node Types:")
    total_nodes = 0
    for ntype in data.node_types:
        n = data[ntype].x.shape[0]
        total_nodes += n
        print(f"    {ntype:<14} : {n:>5} nodes  (feat dim: {data[ntype].x.shape[1]})")

    print("\n  Edge Types:")
    total_edges = 0
    relation_counts = defaultdict(int)
    for (src, rel, dst) in data.edge_types:
        e = data[src, rel, dst].edge_index.shape[1]
        total_edges += e
        relation_counts[rel] += e
        print(f"    {src}-[{rel}]->{dst:<14} : {e:>5} edges")

    print("\n  Degree Statistics:")
    avg_degree = total_edges / max(1, total_nodes)
    print(f"    Overall Avg Degree : {avg_degree:.2f}")
    for rel, count in relation_counts.items():
        avg_rel_degree = count / max(1, total_nodes)
        print(f"    Avg Degree ({rel:<12}) : {avg_rel_degree:.2f}")

    print("\n" + "-" * 60)
    print(f"  Total nodes : {total_nodes}")
    print(f"  Total edges : {total_edges}")
    print("=" * 60 + "\n")
