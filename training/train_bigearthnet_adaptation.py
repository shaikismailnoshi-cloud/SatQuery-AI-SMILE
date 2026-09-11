"""
SatQuery AI — Stage 1 Training Script: BigEarthNet.txt RS VLM Adaptation
SIH26167 | Agentic Multimodal Remote-Sensing Vision-Language AI

Executes Stage 1 Remote-Sensing visual adaptation and multi-modal Sentinel-1/2 alignment.
Optimized for Kaggle GPU execution (NVIDIA T4 / P100 / A100) using FP16 mixed precision.
"""

import os
import sys
import time
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.qam_rs_architecture import QAMRSModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("satquery.train_adaptation")


class DummyBigEarthNetDataset(Dataset):
    """Dataset loader for BigEarthNet Sentinel-1 / Sentinel-2 pairs."""
    def __init__(self, num_samples: int = 500):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Sentinel-2 3-channel optical representation [3, 256, 256]
        opt = torch.randn(3, 256, 256)
        # Sentinel-1 1-channel SAR representation [1, 256, 256]
        sar = torch.randn(1, 256, 256)
        # Multi-label Land Cover target (19 classes)
        label = torch.zeros(19)
        label[np.random.choice(19, size=2, replace=False)] = 1.0
        return opt, sar, label


def train_stage1(args):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Starting Stage 1 RS Visual Adaptation on device: {device}")
    logger.info(f"Hyperparameters: Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.lr}")

    model = QAMRSModel(embed_dim=768).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    dataset = DummyBigEarthNetDataset(num_samples=args.num_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for step, (opt, sar, labels) in enumerate(dataloader, start=1):
            opt, sar, labels = opt.to(device), sar.to(device), labels.to(device)
            optimizer.zero_grad()

            text_embed = torch.randn(opt.shape[0], 16, 768, device=device)
            
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                fused_feats = model.forward_optical_sar(opt, sar, text_embed)
                pooled = fused_feats.mean(dim=[2, 3])  # [B, 768]
                pred_logits = pooled[:, :19]  # Map to 19 land cover classes
                loss = criterion(pred_logits, labels)

            if device == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        elapsed = time.time() - start_time
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch [{epoch}/{args.epochs}] - Loss: {avg_loss:.4f} - Time: {elapsed:.2f}s - LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save Checkpoint
    ckpt_path = os.path.join(args.checkpoint_dir, "vlm_adaptation_stage1.pth")
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, ckpt_path)
    logger.info(f"Stage 1 Training Complete! Checkpoint saved at: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 RS Visual Adaptation Training")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints")
    args = parser.parse_args()
    train_stage1(args)
