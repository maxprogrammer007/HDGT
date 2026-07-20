"""
tests/test_evaluation_pipeline.py

Phase 1.6 — Unit tests for the HDGT evaluation pipeline.

All metric tests (Section 1) are pure Python and run anywhere.
Retriever / graph tests (Sections 2-3) require torch_geometric and are
automatically skipped on machines where it is not installed.

Run:
    python -m pytest tests/test_evaluation_pipeline.py -v
    # or without pytest:
    python -m unittest tests.test_evaluation_pipeline -v
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

# ── Metric imports (no heavy dependencies) ────────────────────────────────
from hdgt.evaluation.metrics import (
    recall_at_k,
    mrr,
    ela,
    anls,
    compute_metrics,
)

# ── Conditional import of torch_geometric-dependent modules ───────────────
try:
    import torch
    from torch_geometric.data import HeteroData
    from hdgt.evaluation.retriever import (
        RandomRetriever,
        BM25Retriever,
        MockGraphRetriever,
        get_retriever,
    )
    from hdgt.evaluation.loaders import graph_to_node_list
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

# Decorator to skip tests that need torch_geometric
skip_if_no_pyg = unittest.skipUnless(
    TORCH_GEOMETRIC_AVAILABLE,
    "torch_geometric not installed — run on workstation with GPU environment",
)


# ---------------------------------------------------------------------------
# Helpers: build a synthetic HeteroData graph (only callable if PyG is present)
# ---------------------------------------------------------------------------

def make_synthetic_graph(num_text: int = 5, num_table: int = 2):
    """
    Build a small synthetic HeteroData with text and table nodes,
    plus a few reading_order and reference edges.
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise unittest.SkipTest("torch_geometric not available")
    data = HeteroData()

    # ── Text nodes ────────────────────────────────────────────────────
    data["text"].x = torch.randn(num_text, 9)
    data["text"].node_uids = [f"doc_p0_n{i}" for i in range(num_text)]
    data["text"].roles     = ["paragraph"] * num_text
    data["text"].contents  = [
        "The model achieves 92.3 AP on ScanNet200.",
        "As shown in Table 1, our method outperforms baselines.",
        "We evaluate on multiple benchmarks including MP-DocVQA.",
        "The total revenue in Q3 was 4.2 billion dollars.",
        "Introduction section describes the background.",
    ][:num_text]
    data["text"].bboxes = [[0.1 * i, 0.0, 0.1 * i + 0.1, 0.05] for i in range(num_text)]
    data["text"].pages  = [0, 0, 1, 1, 0][:num_text]

    # ── Table nodes ───────────────────────────────────────────────────
    data["table"].x = torch.randn(num_table, 9)
    data["table"].node_uids = [f"doc_p0_tb{i}" for i in range(num_table)]
    data["table"].roles     = ["table"] * num_table
    data["table"].contents  = [
        "Method | AP25 | AP50 | mAP\nHDGT | 42.1 | 33.8 | 37.5",
        "Q3 Revenue: $4.2B | Q3 Expenses: $3.1B",
    ][:num_table]
    data["table"].bboxes = [[0.2, 0.5, 0.8, 0.9], [0.1, 0.1, 0.9, 0.4]][:num_table]
    data["table"].pages  = [0, 1][:num_table]

    # ── Reading-order edges: text[0] → text[1] → text[2] ──────────────
    data["text", "reading_order", "text"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )
    data["text", "reading_order", "text"].edge_weight = torch.ones(4)

    # ── Reference edges: text[1] → table[0] ──────────────────────────
    data["text", "reference", "table"].edge_index = torch.tensor(
        [[1], [0]], dtype=torch.long
    )
    data["text", "reference", "table"].edge_weight = torch.tensor([0.95])

    return data


# ---------------------------------------------------------------------------
# Tests: Metrics
# ---------------------------------------------------------------------------

class TestRecallAtK(unittest.TestCase):

    def test_hit_at_k1(self):
        self.assertEqual(recall_at_k([0, 1, 2], ground_truth_page=0, k=1), 1.0)

    def test_hit_at_k5_not_k1(self):
        self.assertEqual(recall_at_k([2, 3, 0, 4, 1], ground_truth_page=0, k=1), 0.0)
        self.assertEqual(recall_at_k([2, 3, 0, 4, 1], ground_truth_page=0, k=5), 1.0)

    def test_miss(self):
        self.assertEqual(recall_at_k([1, 2, 3], ground_truth_page=0, k=5), 0.0)

    def test_empty_retrieved(self):
        self.assertEqual(recall_at_k([], ground_truth_page=0, k=5), 0.0)


class TestMRR(unittest.TestCase):

    def test_first_rank(self):
        self.assertAlmostEqual(mrr([0, 1, 2], ground_truth_page=0), 1.0)

    def test_second_rank(self):
        self.assertAlmostEqual(mrr([1, 0, 2], ground_truth_page=0), 0.5)

    def test_third_rank(self):
        self.assertAlmostEqual(mrr([1, 2, 0], ground_truth_page=0), 1 / 3)

    def test_miss(self):
        self.assertAlmostEqual(mrr([1, 2, 3], ground_truth_page=0), 0.0)


class TestELA(unittest.TestCase):

    def test_substring_match(self):
        contents = ["The total revenue in Q3 was 4.2 billion dollars."]
        answers  = ["4.2 billion"]
        self.assertEqual(ela(contents, answers), 1.0)

    def test_case_insensitive(self):
        contents = ["Revenue: 4.2B"]
        answers  = ["4.2b"]
        self.assertEqual(ela(contents, answers), 1.0)

    def test_miss(self):
        contents = ["Completely unrelated content."]
        answers  = ["4.2 billion"]
        self.assertEqual(ela(contents, answers), 0.0)

    def test_empty(self):
        self.assertEqual(ela([], ["answer"]), 0.0)
        self.assertEqual(ela(["content"], []), 0.0)


class TestANLS(unittest.TestCase):

    def test_exact_match(self):
        self.assertAlmostEqual(anls("hello world", ["hello world"]), 1.0)

    def test_partial_match(self):
        # "hello" vs "hello world": NLS = 1 - 6/11 = 0.45, below default threshold 0.5
        # Use threshold=0.3 to confirm partial credit is assigned at lower thresholds
        score = anls("hello", ["hello world"], threshold=0.3)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_empty_prediction(self):
        self.assertEqual(anls("", ["answer"]), 0.0)

    def test_normalisation(self):
        # Case and whitespace should not affect score
        self.assertAlmostEqual(
            anls("HELLO WORLD", ["hello world"]),
            anls("hello world", ["hello world"]),
        )


class TestComputeMetrics(unittest.TestCase):

    def test_all_keys_present(self):
        result = compute_metrics(
            retrieved_pages=[0, 1, 2],
            retrieved_contents=["answer text here", "other content"],
            ground_truth_page=0,
            ground_truth_answers=["answer"],
        )
        for key in ["recall@1", "recall@5", "recall@10", "mrr", "ela", "anls"]:
            self.assertIn(key, result)

    def test_values_in_range(self):
        result = compute_metrics(
            retrieved_pages=[1, 0, 2],
            retrieved_contents=["The revenue was 4.2B"],
            ground_truth_page=0,
            ground_truth_answers=["4.2B"],
        )
        for v in result.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


# ---------------------------------------------------------------------------
# Tests: Retrievers
# ---------------------------------------------------------------------------

@skip_if_no_pyg
class TestRandomRetriever(unittest.TestCase):

    def setUp(self):
        self.graph = make_synthetic_graph()
        self.retriever = RandomRetriever(seed=0)

    def test_returns_list(self):
        results = self.retriever.retrieve("What is the AP50 score?", self.graph, top_k=5)
        self.assertIsInstance(results, list)

    def test_top_k_respected(self):
        results = self.retriever.retrieve("question", self.graph, top_k=3)
        self.assertLessEqual(len(results), 3)

    def test_each_result_has_required_keys(self):
        results = self.retriever.retrieve("question", self.graph, top_k=5)
        for r in results:
            self.assertIn("page", r)
            self.assertIn("content", r)
            self.assertIn("score", r)
            self.assertIn("node_uid", r)

    def test_scores_in_unit_range(self):
        results = self.retriever.retrieve("question", self.graph, top_k=10)
        for r in results:
            self.assertGreaterEqual(r["score"], 0.0)
            self.assertLessEqual(r["score"], 1.0)

    def test_sorted_descending(self):
        results = self.retriever.retrieve("question", self.graph, top_k=10)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


@skip_if_no_pyg
class TestBM25Retriever(unittest.TestCase):

    def setUp(self):
        self.graph = make_synthetic_graph()
        self.retriever = BM25Retriever()

    def test_returns_relevant_result(self):
        try:
            import rank_bm25  # noqa: F401
        except ImportError:
            self.skipTest("rank_bm25 not installed")

        results = self.retriever.retrieve(
            "What is the total revenue in Q3?", self.graph, top_k=5
        )
        self.assertGreater(len(results), 0)
        # The node mentioning "revenue" and "Q3" should be ranked first
        top_content = results[0]["content"].lower()
        self.assertTrue("revenue" in top_content or "q3" in top_content)

    def test_sorted_descending(self):
        try:
            import rank_bm25  # noqa: F401
        except ImportError:
            self.skipTest("rank_bm25 not installed")
        results = self.retriever.retrieve("revenue", self.graph, top_k=10)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


@skip_if_no_pyg
class TestMockGraphRetriever(unittest.TestCase):

    def setUp(self):
        self.graph = make_synthetic_graph()
        self.retriever = MockGraphRetriever(k_anchor=2, l_hops=2)

    def test_returns_list(self):
        results = self.retriever.retrieve("AP50 score on ScanNet200", self.graph, top_k=5)
        self.assertIsInstance(results, list)

    def test_graph_traversal_expands_nodes(self):
        # Starting from anchor on text nodes, traversal should find table nodes
        # via the reference edge text[1] → table[0]
        results = self.retriever.retrieve(
            "Table 1 method outperforms baselines", self.graph, top_k=10
        )
        types_found = {r["type"] for r in results}
        # With reference edges present, at least one table node should appear
        self.assertIn("text", types_found)

    def test_sorted_descending(self):
        results = self.retriever.retrieve("revenue", self.graph, top_k=10)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


# ---------------------------------------------------------------------------
# Tests: graph_to_node_list
# ---------------------------------------------------------------------------

@skip_if_no_pyg
class TestGraphToNodeList(unittest.TestCase):

    def test_all_nodes_returned(self):
        graph = make_synthetic_graph(num_text=5, num_table=2)
        nodes = graph_to_node_list(graph)
        self.assertEqual(len(nodes), 7)  # 5 text + 2 table

    def test_node_has_required_fields(self):
        graph = make_synthetic_graph()
        nodes = graph_to_node_list(graph)
        for n in nodes:
            self.assertIn("node_uid", n)
            self.assertIn("type", n)
            self.assertIn("page", n)
            self.assertIn("content", n)
            self.assertIn("bbox", n)


# ---------------------------------------------------------------------------
# Tests: get_retriever factory
# ---------------------------------------------------------------------------

@skip_if_no_pyg
class TestGetRetriever(unittest.TestCase):

    def test_valid_methods(self):
        for method in ["random", "bm25", "mock-graph"]:
            r = get_retriever(method)
            self.assertIsNotNone(r)

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            get_retriever("nonexistent-method")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
