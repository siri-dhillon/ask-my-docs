Passage-level (the retrieved chunk must contain the answer evidence):

| Mode | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | Doc Recall@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| Vector only | 0.533 | 0.822 | 0.844 | 0.956 | 0.694 | 1.000 | 3 | 4 |
| Hybrid (BM25 + dense, RRF) | 0.533 | 0.844 | 0.867 | 0.978 | 0.705 | 1.000 | 13 | 16 |
| Hybrid + rerank | 0.667 | 0.956 | 0.978 | 1.000 | 0.808 | 1.000 | 1120 | 1284 |
