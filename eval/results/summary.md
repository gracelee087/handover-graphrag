# Evaluation: 40 questions x 3 retrievers

Model `claude-opus-5-5`, embeddings `all-MiniLM-L6-v2`, store `data/handover.rl` (Relate).

Judge verdicts per question type (correct / partial / mean score where correct=1, partial=0.5).

| question type | n | vector correct / partial / score | graph correct / partial / score | text2cypher correct / partial / score |
|---|---|---|---|---|
| factual | 8 | 4 / 0 / 0.50 | 8 / 0 / 1.00 | 8 / 0 / 1.00 |
| aggregate | 8 | 1 / 2 / 0.25 | 1 / 2 / 0.25 | 7 / 0 / 0.88 |
| multihop | 8 | 6 / 1 / 0.81 | 7 / 1 / 0.94 | 7 / 1 / 0.94 |
| compliance | 8 | 1 / 5 / 0.44 | 5 / 3 / 0.81 | 2 / 4 / 0.50 |
| handover | 8 | 3 / 2 / 0.50 | 4 / 4 / 0.75 | 5 / 3 / 0.81 |
| ALL | 40 | 15 / 10 / 0.50 | 25 / 10 / 0.75 | 29 / 8 / 0.82 |

Answering cost (list price): USD 3.75 for 120 answers; judge: USD 0.72.
Mean latency per answer (s): {'vector': 4.5, 'graph': 5.3, 'text2cypher': 10.8}
