"""
hdgt/evaluation/__init__.py

Phase 1.6 evaluation subpackage.

Modules:
  loaders   — MPDocVQALoader, GraphLoader (requires torch_geometric)
  metrics   — Recall@K, MRR, ELA, ANLS   (pure Python, no GPU needed)
  retriever — RandomRetriever, BM25Retriever, MockGraphRetriever (requires torch_geometric)

Imports are lazy so that metrics can be used without torch_geometric installed.
"""

# Always-available pure-Python metrics
from hdgt.evaluation.metrics import (
    recall_at_k,
    mrr,
    ela,
    anls,
    compute_metrics,
)

# torch_geometric-dependent modules — imported lazily
def _load_pyg_modules():
    from hdgt.evaluation.loaders import MPDocVQALoader, GraphLoader, graph_to_node_list
    from hdgt.evaluation.retriever import (
        RandomRetriever,
        BM25Retriever,
        MockGraphRetriever,
        get_retriever,
    )
    return (
        MPDocVQALoader, GraphLoader, graph_to_node_list,
        RandomRetriever, BM25Retriever, MockGraphRetriever, get_retriever,
    )


__all__ = [
    # Metrics (always available)
    "recall_at_k",
    "mrr",
    "ela",
    "anls",
    "compute_metrics",
    # Loaders & retrievers (require torch_geometric)
    "MPDocVQALoader",
    "GraphLoader",
    "graph_to_node_list",
    "RandomRetriever",
    "BM25Retriever",
    "MockGraphRetriever",
    "get_retriever",
]
