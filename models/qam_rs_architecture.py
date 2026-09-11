"""
SatQuery AI — QAM-RS Model Architecture
Query-Adaptive Multimodal Remote-Sensing Intelligence System
SIH26167 | Agentic Multimodal Remote-Sensing Vision-Language AI

Modular PyTorch implementation featuring RS vision encoders, query-conditioned cross-attention,
bi-temporal change encoders, optical-SAR gated fusion, visual grounding heads, and VQA heads.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List


class RSVisionEncoder(nn.Module):
    """
    Remote-Sensing Vision Encoder supporting arbitrary band counts (Optical RGB, Sentinel-2 12-band, SAR).
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Band Projection Layer
        self.band_proj = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Feature Layers
        self.stage1 = self._make_layer(64, 128, stride=2)
        self.stage2 = self._make_layer(128, 256, stride=2)
        self.stage3 = self._make_layer(256, embed_dim, stride=2)

    def _make_layer(self, in_c: int, out_c: int, stride: int = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, C, H, W]
        x = self.band_proj(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)  # [B, embed_dim, H', W']
        return x


class QueryConditionedFusion(nn.Module):
    """
    Injects text query embeddings into visual feature maps using Cross-Attention.
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, visual_feats: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        # visual_feats: [B, C, H, W] -> flatten to [B, N, C]
        B, C, H, W = visual_feats.shape
        v_flat = visual_feats.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Cross attention: Query = text_embeds, Key = v_flat, Value = v_flat
        attn_out, _ = self.cross_attn(query=v_flat, key=text_embeds, value=text_embeds)
        fused = self.norm(v_flat + attn_out)

        # Reshape back to [B, C, H, W]
        fused_spatial = fused.transpose(1, 2).view(B, C, H, W)
        return fused_spatial


class BiTemporalEncoder(nn.Module):
    """
    Extracts T1, T2 representations, calculates difference features,
    and applies temporal cross-attention for change reasoning.
    """

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.encoder = RSVisionEncoder(in_channels=3, embed_dim=embed_dim)
        self.diff_conv = nn.Sequential(
            nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, img_t1: torch.Tensor, img_t2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f_t1 = self.encoder(img_t1)
        f_t2 = self.encoder(img_t2)
        diff = torch.abs(f_t2 - f_t1)

        # Concat T1, T2, and Difference features
        concat_feats = torch.cat([f_t1, f_t2, diff], dim=1)
        change_feats = self.diff_conv(concat_feats)
        return change_feats, diff


class GatedCrossModalAttention(nn.Module):
    """
    Fuses Optical and SAR features using learned gating logic.
    """

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.gate_layer = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, opt_feats: torch.Tensor, sar_feats: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([opt_feats, sar_feats], dim=1)
        gate = self.gate_layer(combined)
        gated_fusion = gate * opt_feats + (1.0 - gate) * sar_feats
        return self.fusion_conv(gated_fusion)


class GroundingHead(nn.Module):
    """
    Predicts normalized bounding box target coordinates [ymin, xmin, ymax, xmax].
    """

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bbox_regressor = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def forward(self, fused_feats: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(fused_feats).flatten(1)
        bboxes = self.bbox_regressor(pooled)
        return bboxes


class ChangeMaskHead(nn.Module):
    """
    Generates 2D binary change probability map.
    """

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, change_feats: torch.Tensor) -> torch.Tensor:
        return self.decoder(change_feats)


class QAMRSModel(nn.Module):
    """
    Complete QAM-RS Unified Multimodal Model Architecture.
    """

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim
        self.optical_encoder = RSVisionEncoder(in_channels=3, embed_dim=embed_dim)
        self.sar_encoder = RSVisionEncoder(in_channels=1, embed_dim=embed_dim)
        self.temporal_encoder = BiTemporalEncoder(embed_dim=embed_dim)
        self.query_fusion = QueryConditionedFusion(embed_dim=embed_dim)
        self.gated_fusion = GatedCrossModalAttention(embed_dim=embed_dim)
        
        # Heads
        self.grounding_head = GroundingHead(embed_dim=embed_dim)
        self.change_mask_head = ChangeMaskHead(embed_dim=embed_dim)

    def forward_single_optical(self, img: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        v_feats = self.optical_encoder(img)
        fused = self.query_fusion(v_feats, text_embeds)
        return fused

    def forward_optical_sar(self, opt_img: torch.Tensor, sar_img: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        opt_feats = self.optical_encoder(opt_img)
        sar_feats = self.sar_encoder(sar_img)
        fused_modality = self.gated_fusion(opt_feats, sar_feats)
        fused = self.query_fusion(fused_modality, text_embeds)
        return fused

    def forward_bitemporal(self, img_t1: torch.Tensor, img_t2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        change_feats, diff = self.temporal_encoder(img_t1, img_t2)
        change_mask = self.change_mask_head(change_feats)
        return change_feats, change_mask
