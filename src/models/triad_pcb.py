from __future__ import annotations
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .clip_backbone import CLIPLikeBackbone
from .projection_head import ProjectionHead
from .text_branch import TextBranch
from .online_branch import OnlineBranch
from .reference_branch import ReferenceBranch
from .fusion import fuse_anomaly_maps

class TriADPCB(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        patch_size: int = 16,
        num_anchors: int = 8,
        num_abnormal_prompts: int = 4,
        hist_capacity: int = 50,
        sample_ratio: float = 0.5,
        top_fraction: float = 0.3,
        gate_threshold: float = 0.5,
        fusion_temperature: float = 0.5,
        residual_weight: float = 0.5,
    ):
        super().__init__()
        self.backbone = CLIPLikeBackbone(embed_dim=embed_dim, patch_size=patch_size)
        self.projector = ProjectionHead(in_dim=embed_dim, out_dim=embed_dim)
        self.text_branch = TextBranch(
            text_encoder=self.backbone.text_encoder,
            dim=embed_dim,
            num_anchors=num_anchors,
            num_abnormal_prompts=num_abnormal_prompts,
        )
        self.online_branch = OnlineBranch(
            capacity=hist_capacity,
            sample_ratio=sample_ratio,
            top_fraction=top_fraction,
            gate_threshold=gate_threshold,
        )
        self.reference_branch = ReferenceBranch(
            sample_ratio=sample_ratio,
            min_rep=32,
        )
        self.fusion_temperature = fusion_temperature
        self.residual_weight = residual_weight

    def encode(self, images: torch.Tensor):
        patch_feats, global_feat, grid = self.backbone.encode_image(images)
        z_patch, z_global = self.projector(patch_feats, global_feat)
        return z_patch, z_global, grid

    def build_reference_bank(self, ref_images: torch.Tensor) -> None:
        self.reference_branch.eval()
        with torch.no_grad():
            z_patch, _, _ = self.encode(ref_images)
            feats = [z_patch[i] for i in range(z_patch.size(0))]
            self.reference_branch.build_bank(feats)

    def reset_stream_state(self) -> None:
        self.online_branch.reset()

    def forward(self, images: torch.Tensor) -> dict:
        z_patch, z_global, grid = self.encode(images)
        tb = self.text_branch(z_patch, z_global)
        hb = self.online_branch.infer(z_patch)
        rb = self.reference_branch.infer(z_patch)
        fused = fuse_anomaly_maps(
            tb["anomaly_map"],
            hb,
            rb,
            hist_conf=self.online_branch.confidence(),
            ref_conf=self.reference_branch.confidence(),
            fusion_temperature=self.fusion_temperature,
            residual_weight=self.residual_weight,
        )
        for i in range(images.size(0)):
            self.online_branch.update(z_patch[i:i+1], fused["image_score"][i])
        return {
            "text_map": tb["anomaly_map"],
            "hist_map": hb,
            "ref_map": rb,
            **fused,
            "anchor_weights": tb["anchor_weights"],
            "pseudo_normal": tb["pseudo_normal"],
            "pseudo_abnormal": tb["pseudo_abnormal"],
            "patch_feats": z_patch,
            "global_feat": z_global,
            "grid": grid,
        }

    def training_losses(
        self,
        images: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        masks_available: bool = False,
        lambda_nce: float = 1.0,
        lambda_abn: float = 1.0,
        lambda_anc: float = 0.1,
    ) -> dict:
        from ..losses.segmentation_loss import segmentation_loss
        from ..losses.contrastive_loss import image_conditioned_contrastive_loss
        from ..losses.abnormality_loss import abnormality_loss

        z_patch, z_global, grid = self.encode(images)
        tb = self.text_branch(z_patch, z_global)
        losses = {}
        if masks_available and masks is not None:
            amap = tb["anomaly_map"].unsqueeze(1)
            amap = F.interpolate(amap, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            losses["seg"] = segmentation_loss(amap, masks)
        else:
            losses["seg"] = torch.tensor(0.0, device=images.device)

        positive_prompt = torch.einsum("bk,kd->bd", tb["anchor_weights"], self.text_branch.normal_prompts)
        negative_bank = self.text_branch.abnormal_prompts.reshape(-1, self.text_branch.abnormal_prompts.size(-1))
        losses["nce"] = image_conditioned_contrastive_loss(z_patch, positive_prompt, negative_bank)
        losses["abn"] = abnormality_loss(
            tb["pseudo_normal"].view(images.size(0), -1, z_patch.size(-1)),
            self.text_branch.normal_prompts,
            tb["pseudo_abnormal"].view(images.size(0), -1, z_patch.size(-1)),
            self.text_branch.abnormal_prompts,
        )
        losses["anc"] = self.text_branch.regularization_loss(self.text_branch.anchor_vectors.detach())
        losses["total"] = losses["seg"] + lambda_nce * losses["nce"] + lambda_abn * losses["abn"] + lambda_anc * losses["anc"]
        return losses
