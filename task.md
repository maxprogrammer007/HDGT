# HDGT — Task Tracker & Project Roadmap

## Current Phase: 1.6 — MP-DocVQA Benchmark Integration & Evaluation Infrastructure

---

## 📊 Component Assessment

| Component | Status | Details |
| :--- | :---: | :--- |
| **Phase 1 Graph Construction** | ✅ 100% | Docling parser, node/edge builders, PyG `HeteroData` assembly |
| **Phase 1.5 Semantic Validation** | ✅ 100% | Edge connectivity, spatial/hierarchy topology validation |
| **MP-DocVQA Dataset Integration** | ✅ 100% | 8,989 multi-page PDFs compiled; QA context mapping complete |
| **Evaluation Framework** | ✅ 100% | ANLS, Recall@K, MRR, ELA, `MPDocVQALoader`, `GraphLoader` |
| **Unit Testing** | ✅ 100% | 32/32 tests passing (`tests/test_evaluation_pipeline.py`) |
| **Full Benchmark Evaluation** | ⏳ In Progress | Parallel graph building (`--num-workers 8`) + full-val BM25 baseline |
| **Literature Comparison** | ✅ Complete | SOTA baseline matrix documented in `phase1_6_results/baseline_comparison.md` |
| **Phase 2 Semantic Encoding** | ⏳ Pending | Qwen2.5-VL node embeddings + GNN Encoder (blocked until Phase 1.6 completes) |

---

## 🗺️ Project Roadmap

```text
Phase 1: PDF → Multi-Relational Heterogeneous Graph
  └── ✅ Complete

Phase 1.5: Graph Topology & Semantic Relationship Validation
  └── ✅ Complete

Phase 1.6: MP-DocVQA Integration & Evaluation Framework
  ├── ✓ Dataset extraction & PDF compilation (8,989 PDFs)
  ├── ✓ Evaluation pipeline (ANLS, Recall@K, MRR, ELA)
  ├── ✓ Data & Graph Loaders
  ├── ✓ Unit tests (32/32 passing)
  ├── ⏳ Parallel graph generation over val split (In Progress)
  ├── ⏳ Full validation split BM25 baseline evaluation
  └── ✓ SOTA literature comparison table

Phase 2: Vision-Language Node Encoding & Token Compression
  ├── ⏳ Qwen2.5-VL multimodal backbone integration
  ├── ⏳ High-resolution DocCompressor (layout-aware token compression)
  └── ⏳ Node feature matrix population (data[node_type].x)
```

---

## 🎯 Immediate Action Items

- [x] Adopt scientific framing: distinguish evaluation infrastructure validation from model performance.
- [x] Restructure graph builder (`build_mpdocvqa_graphs.py`) for multi-worker parallel execution (8 processes across 16 CPU cores).
- [x] Install `rank-bm25` and ensure 32/32 unit tests pass.
- [ ] Complete graph construction for full validation set (1,329 contexts).
- [ ] Compute full validation set BM25 baseline metrics (`evaluate_retrieval.py --split val --method bm25`).
- [ ] Log graph construction statistics (documents processed, success/fail rate, avg nodes/edges per graph, avg build time).
