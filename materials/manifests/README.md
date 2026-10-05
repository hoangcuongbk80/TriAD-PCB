# Dataset manifests

Create one CSV per dataset with the columns below:

```text
image_path,mask_path,label,split,category,source_id
```

Expected manuscript counts:

| Dataset | Optimization normal | Validation normal | Test normal | Test anomalous |
|---|---:|---:|---:|---:|
| DsPCBSD+ | 8,100 | 900 | 809 | 450 |
| TDD-Net PCB | 8,928 | 992 | 1,502 | 1,006 |
| VisA PCB | 3,150 | 350 | 516 | 400 |

Run both checks before training:

```powershell
python scripts/validate_manifest.py <manifest.csv> --checksums <dataset>_files.csv
python scripts/audit_manifest_counts.py <manifest.csv> --dataset "<dataset name>"
python scripts/summarize_manifest.py <manifest.csv> --output <dataset>_summary.json
```

Use the exact dataset names `DsPCBSD+`, `TDD-Net PCB`, and `VisA PCB` for the count audit. Every crop from the same source image or board must share one globally unique `source_id`. The final dataset manifests are derived artifacts and require the corresponding dataset images and released evaluation masks, which are not stored in this workspace.

