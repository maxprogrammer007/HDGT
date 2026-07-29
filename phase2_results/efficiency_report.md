# HDGT Pipeline Efficiency Report

Measured on 20 validation graphs.

| Component | Mean (ms) | ± Std | Notes |
| :--- | :---: | :---: | :--- |
| Graph loading (disk I/O) | 8.2 | 18.8 | per document |
| BM25 index + query | 1.3 | 1.0 | per document |
| Qwen feat → GPU transfer | 0.8 | 2.4 | ~69 nodes avg |
| GNN forward pass | 103.1 | 436.2 | 1493.54 μs/node |
| Cosine ranking | 27.8 | 119.3 | 403.27 μs/node |
| TOTAL (end-to-end) | 141.2 | 452.6 | per query |

- **GPU Memory**: 13.8 MB
- **Avg nodes/graph**: 69
- **Device**: cuda:0
