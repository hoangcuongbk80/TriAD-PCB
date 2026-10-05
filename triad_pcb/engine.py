from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .fusion import confidence_weights, fuse_maps, top_q_pool
from .memory import HistoricalBank, ReferenceBank


class CausalInferenceSession:
    """Stateful evaluator that scores first and updates history second."""

    def __init__(self, model, config: dict):
        self.model = model
        self.config = config
        memory_cfg = config["memory"]
        self.reference = ReferenceBank(
            memory_cfg["sampling_ratio"], memory_cfg["minimum_reference_patches"]
        )
        self.history = HistoricalBank(
            memory_cfg["history_capacity"],
            memory_cfg["sampling_ratio"],
            memory_cfg["history_top_ratio"],
        )
        branches = config.get("evaluation", {}).get(
            "branches", ["text", "online", "reference"]
        )
        allowed = {"text", "online", "reference"}
        unknown = set(branches).difference(allowed)
        if unknown or not branches:
            raise ValueError(f"Invalid active branches: {sorted(unknown) if unknown else branches}")
        self.active_branches = set(branches)

    @torch.no_grad()
    def initialize_support(self, support_images: torch.Tensor) -> None:
        self.model.eval()
        if "reference" not in self.active_branches:
            return
        patch_batches, _ = self.model.encode_image(support_images)
        self.reference.build([patches for patches in patch_batches])

    @torch.no_grad()
    def step(self, image: torch.Tensor) -> dict[str, torch.Tensor | float | bool]:
        if image.shape[0] != 1:
            raise ValueError("Causal inference processes one image at a time")
        patches, global_features = self.model.encode_image(image)
        query = patches[0]
        text_map = (
            self.model.text_map(patches, global_features)[0][0]
            if "text" in self.active_branches
            else torch.zeros(query.shape[0], device=query.device, dtype=query.dtype)
        )
        history_map = (
            self.history.score(query)
            if "online" in self.active_branches
            else torch.zeros(query.shape[0], device=query.device, dtype=query.dtype)
        )
        reference_map = (
            self.reference.score(query)
            if "reference" in self.active_branches
            else torch.zeros(query.shape[0], device=query.device, dtype=query.dtype)
        )
        model_cfg = self.config["model"]
        weights = confidence_weights(
            self.history.confidence,
            self.reference.confidence,
            float(model_cfg["fusion_temperature"]),
            device=query.device,
            active=(
                "text" in self.active_branches,
                "online" in self.active_branches,
                "reference" in self.active_branches,
            ),
        )
        final_patch_map = fuse_maps(
            text_map,
            history_map,
            reference_map,
            weights,
            float(model_cfg["residual_weight"]),
        )
        image_score = top_q_pool(final_patch_map, float(model_cfg["top_q"]))
        accepted = bool(image_score.item() < float(self.config["memory"]["normality_threshold"]))
        # This update is deliberately after all maps and the image score are computed.
        if accepted and "online" in self.active_branches:
            self.history.update(query)
        side = int(math.sqrt(final_patch_map.numel()))
        if side * side != final_patch_map.numel():
            raise ValueError("Patch token count is not a square grid")
        pixel_map = F.interpolate(
            final_patch_map.reshape(1, 1, side, side),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return {
            "image_score": float(image_score.item()),
            "pixel_map": pixel_map,
            "patch_map": final_patch_map,
            "fusion_weights": weights,
            "accepted_into_history": accepted,
            "history_size_after_update": len(self.history),
        }
