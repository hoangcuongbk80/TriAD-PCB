# Anchor relevance scores

Run `triad-select-anchors` separately for DsPCBSD+, TDD-Net PCB, and VisA PCB. Store the outputs in this directory as `dspcbsd.csv`, `tdd.csv`, and `visa.csv`, then run:

```powershell
python scripts/merge_anchor_scores.py materials/anchor_scores/dspcbsd.csv materials/anchor_scores/tdd.csv materials/anchor_scores/visa.csv --output materials/anchor_scores/equal_weight_ranking.csv
```

The score files are dataset-dependent derived artifacts and cannot be generated without the normal optimization images. Each output records the candidate, rank, and mean cosine relevance. The merged file uses an equal arithmetic mean across the three dataset-specific scores.
