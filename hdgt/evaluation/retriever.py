"""
hdgt/evaluation/retriever.py

Phase 1.6 — Baseline retrievers for HDGT retrieval evaluation.

All retrievers share the same interface:
    retriever.retrieve(question, graph) -> List[dict]

Each result dict contains:
    page    : int   — 0-indexed page number of the node
    content : str   — raw text content of the node
    score   : float — retrieval score (higher = better)
    node_uid: str   — globally unique node identifier
    type    : str   — node type (text, table, figure, section, page)

Classes
-------
RandomRetriever      — Random baseline (score = U[0,1])
BM25Retriever        — Sparse text retrieval using BM25 (rank_bm25)
MockGraphRetriever   — Simulates Phase 3 graph traversal using character overlap
                       for anchor scoring + l-hop BFS via structural edges
"""

from __future__ import annotations

import logging
import random
import re
from collections import deque
from typing import Any, Dict, List, Optional

from torch_geometric.data import HeteroData

from hdgt.evaluation.loaders import graph_to_node_list

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared interface
# ---------------------------------------------------------------------------

class BaseRetriever:
    """Abstract base class for all retrievers."""

    name: str = "base"

    def retrieve(
        self,
        question: str,
        graph: HeteroData,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and rank nodes from `graph` relevant to `question`.

        Parameters
        ----------
        question : str
        graph    : HeteroData — must have .contents, .pages, .node_uids on node types
        top_k    : int        — maximum number of results to return

        Returns
        -------
        List of dicts, sorted by descending score, length <= top_k.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# RandomRetriever
# ---------------------------------------------------------------------------

class RandomRetriever(BaseRetriever):
    """
    Assigns a uniform random score to every node. Used as lower-bound baseline.

    Parameters
    ----------
    seed : int | None
        Random seed for reproducibility. None = non-deterministic.
    """

    name = "random"

    def __init__(self, seed: Optional[int] = 42) -> None:
        self._rng = random.Random(seed)

    def retrieve(
        self,
        question: str,
        graph: HeteroData,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        nodes = graph_to_node_list(graph)
        for node in nodes:
            node["score"] = self._rng.random()
        nodes.sort(key=lambda n: n["score"], reverse=True)
        return nodes[:top_k]


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------

class BM25Retriever(BaseRetriever):
    """
    BM25 sparse text retrieval over node contents.

    Requires `rank_bm25` (pip install rank-bm25).
    Falls back gracefully if the library is not installed.

    The entire graph is tokenised at construction time (per-graph caching).
    The same instance can be reused across graphs because `retrieve()` always
    re-indexes the provided graph.

    Parameters
    ----------
    tokeniser : callable | None
        Custom tokeniser function str -> List[str].
        Default: lowercase whitespace split.
    """

    name = "bm25"

    def __init__(self, tokeniser=None) -> None:
        self._tokeniser = tokeniser or (lambda s: re.sub(r"[^\w\s]", " ", s.lower()).split())

    def retrieve(
        self,
        question: str,
        graph: HeteroData,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.error(
                "rank_bm25 is not installed. Run: pip install rank-bm25\n"
                "Falling back to RandomRetriever."
            )
            return RandomRetriever().retrieve(question, graph, top_k)

        nodes = graph_to_node_list(graph)
        if not nodes:
            return []

        # Tokenise corpus
        corpus = [self._tokeniser(n["content"]) for n in nodes]
        bm25 = BM25Okapi(corpus)

        # Score against query
        query_tokens = self._tokeniser(question)
        scores = bm25.get_scores(query_tokens)

        for i, node in enumerate(nodes):
            node["score"] = float(scores[i])

        nodes.sort(key=lambda n: n["score"], reverse=True)
        return nodes[:top_k]


# ---------------------------------------------------------------------------
# MockGraphRetriever
# ---------------------------------------------------------------------------

class MockGraphRetriever(BaseRetriever):
    """
    Simulates the Phase 3 HDGT graph traversal without VLM embeddings.

    Algorithm
    ---------
    1. Score each node by character-level overlap between node content
       and the question (proxy for embedding cosine similarity).
    2. Select the top `k_anchor` nodes as anchors.
    3. Perform l-hop BFS from each anchor following structural edges
       (reading_order, parent_child, reference, continuation).
    4. Combine: anchor score + hop decay (0.9^hop_distance).
    5. Return top-K nodes sorted by combined score.

    Edge priority for traversal:
        reference    → 1.0  (highest priority — evidence links)
        parent_child → 0.9
        reading_order→ 0.7
        continuation → 0.8
        spatial      → 0.5  (last resort)
        contains     → 0.4  (page membership only)

    Parameters
    ----------
    k_anchor : int
        Number of anchor nodes to start BFS from.
    l_hops : int
        Maximum BFS depth.
    hop_decay : float
        Score multiplier per hop (0 < hop_decay < 1).
    """

    name = "mock-graph"

    # Traversal priority weights for each relation type
    EDGE_PRIORITY: Dict[str, float] = {
        "reference":     1.0,
        "parent_child":  0.9,
        "continuation":  0.8,
        "reading_order": 0.7,
        "spatial":       0.5,
        "contains":      0.4,
    }

    def __init__(
        self,
        k_anchor: int = 3,
        l_hops: int = 2,
        hop_decay: float = 0.9,
    ) -> None:
        self.k_anchor  = k_anchor
        self.l_hops    = l_hops
        self.hop_decay = hop_decay

    # ------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        graph: HeteroData,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        nodes = graph_to_node_list(graph)
        if not nodes:
            return []

        # Step 1: score by character overlap (proxy for embedding sim)
        anchor_scores = self._char_overlap_scores(question, nodes)

        # Build a uid → node dict for fast lookup
        uid_to_node = {n["node_uid"]: n for n in nodes}

        # Step 2: select top anchor nodes
        ranked = sorted(
            zip(anchor_scores, nodes),
            key=lambda x: x[0],
            reverse=True,
        )
        anchors = [node for _, node in ranked[: self.k_anchor]]

        # Step 3: build adjacency list from edge_index tensors
        adj = self._build_adjacency(graph, uid_to_node)

        # Step 4: BFS from each anchor
        visited_scores: Dict[str, float] = {}

        for anchor_score, anchor in zip(
            [s for s, _ in ranked[: self.k_anchor]], anchors
        ):
            uid = anchor["node_uid"]
            queue: deque = deque()
            queue.append((uid, 0, anchor_score))

            while queue:
                cur_uid, hop, cur_score = queue.popleft()

                if hop > self.l_hops:
                    continue

                # Record best score seen for this node
                if cur_uid not in visited_scores or visited_scores[cur_uid] < cur_score:
                    visited_scores[cur_uid] = cur_score

                if hop == self.l_hops:
                    continue

                # Expand neighbours
                for neighbour_uid, edge_priority in adj.get(cur_uid, []):
                    next_score = cur_score * self.hop_decay * edge_priority
                    if (
                        neighbour_uid not in visited_scores
                        or visited_scores[neighbour_uid] < next_score
                    ):
                        queue.append((neighbour_uid, hop + 1, next_score))

        # Step 5: build result list
        results = []
        for uid, score in visited_scores.items():
            if uid in uid_to_node:
                node = uid_to_node[uid].copy()
                node["score"] = score
                results.append(node)

        results.sort(key=lambda n: n["score"], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _char_overlap_scores(question: str, nodes: List[dict]) -> List[float]:
        """
        Character-level Jaccard overlap between question tokens and node content.
        Used as a proxy for embedding cosine similarity in Phase 1.6.
        """
        q_tokens = set(re.sub(r"[^\w\s]", " ", question.lower()).split())
        scores = []
        for node in nodes:
            n_tokens = set(re.sub(r"[^\w\s]", " ", node["content"].lower()).split())
            if not q_tokens and not n_tokens:
                scores.append(0.0)
            elif not q_tokens or not n_tokens:
                scores.append(0.0)
            else:
                intersection = len(q_tokens & n_tokens)
                union = len(q_tokens | n_tokens)
                scores.append(intersection / union)
        return scores

    def _build_adjacency(
        self,
        graph: HeteroData,
        uid_to_node: Dict[str, dict],
    ) -> Dict[str, List[tuple]]:
        """
        Build {src_uid: [(dst_uid, edge_priority), ...]} adjacency list
        from the HeteroData edge_index tensors.
        """
        adj: Dict[str, List[tuple]] = {}

        for (src_type, relation, dst_type) in graph.edge_types:
            priority = self.EDGE_PRIORITY.get(relation, 0.5)
            edge_data = graph[src_type, relation, dst_type]
            edge_index = edge_data.edge_index  # (2, E)

            src_uids = getattr(graph[src_type], "node_uids", [])
            dst_uids = getattr(graph[dst_type], "node_uids", [])

            if not src_uids or not dst_uids:
                continue

            num_edges = edge_index.shape[1]
            for e in range(num_edges):
                src_local = int(edge_index[0, e])
                dst_local = int(edge_index[1, e])
                if src_local >= len(src_uids) or dst_local >= len(dst_uids):
                    continue

                src_uid = src_uids[src_local]
                dst_uid = dst_uids[dst_local]

                if src_uid not in adj:
                    adj[src_uid] = []
                adj[src_uid].append((dst_uid, priority))

        return adj


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_retriever(method: str, **kwargs) -> BaseRetriever:
    """
    Instantiate a retriever by name.

    Parameters
    ----------
    method : str
        One of 'random', 'bm25', 'mock-graph'.

    Returns
    -------
    BaseRetriever instance.
    """
    registry = {
        "random":     RandomRetriever,
        "bm25":       BM25Retriever,
        "mock-graph": MockGraphRetriever,
    }
    if method not in registry:
        raise ValueError(
            f"Unknown retrieval method {method!r}. "
            f"Choose from: {list(registry.keys())}"
        )
    return registry[method](**kwargs)
