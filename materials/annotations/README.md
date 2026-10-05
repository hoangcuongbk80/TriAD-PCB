# Evaluation-mask release

The release archive must contain the 450 DsPCBSD+ and 1,006 TDD-Net PCB anomalous-image masks described in the manuscript. Masks are not reconstructed from bounding boxes. Each file must be the reviewed binary polygon raster associated with the corresponding evaluation image.

Before packaging, place the masks under a release root and prepare `metadata.csv` with these columns:

```text
dataset,image_id,mask_path,annotator_count,review_status
```

The `review_status` field should identify whether the mask was included in the three-annotator agreement subset, cross-reviewed, resolved by consensus, or adjudicated. Supply the distribution license or permission notice as a separate text file. Then run `scripts/package_annotations.py`. The script validates paths, non-empty anomalous masks, exact dataset counts, and writes an archive containing metadata, license information, and SHA-256 checksums.

The actual annotation files and their license are external assets and are not present in this workspace.
