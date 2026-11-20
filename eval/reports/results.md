Passage-level (the retrieved chunk must contain the answer evidence):

| Mode | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | Doc Recall@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| Vector only | 0.533 | 0.822 | 0.844 | 0.956 | 0.692 | 1.000 | 3 | 4 |
| Hybrid (BM25 + dense, RRF) | 0.533 | 0.844 | 0.867 | 0.978 | 0.707 | 1.000 | 14 | 17 |
| Hybrid + rerank | 0.667 | 0.933 | 0.956 | 1.000 | 0.803 | 1.000 | 1218 | 1451 |
