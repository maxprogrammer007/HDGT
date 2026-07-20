"""
hdgt/evaluation/__init__.py

Phase 1.6 evaluation subpackage.

Modules:
  loaders   — MPDocVQALoader, GraphLoader
  metrics   — Recall@K, MRR, ELA, ANLS
  retriever — RandomRetriever, BM25Retriever, MockGraphRetriever
"""
from hdgt.evaluation.loaders import MPDocVQALoader, GraphLoader
from hdgt.evaluation.metrics import recall_at_k, mrr, ela, anls
from hdgt.evaluation.retriever import RandomRetriever, BM25Retriever, MockGraphRetriever

__all__ = [
    "MPDocVQALoader",
    "GraphLoader",
    "recall_at_k",
    "mrr",
    "ela",
    "anls",
    "RandomRetriever",
    "BM25Retriever",
    "MockGraphRetriever",
]
