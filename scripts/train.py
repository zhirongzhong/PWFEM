from typing import *
#!/usr/bin/env python3
"""
Train script for ProbSR (Wavelet + Student-t prior + SVGD)

Usage:
  python scripts/train.py --train_dir data/train --val_dir data/test --epochs 50 --batch 4 --particles 50
"""

import argparse, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.ndimage import zoom
from sklearn.metrics import mean_squared_error

from probsr.dataset import NPZDataset
from probsr.models import WaveletModule, DownscaleResNet
from probsr.svgd import svgd_step
from probsr.utils import to_tensor_img

# ---------- Prior ----------
def grad_logp_student_t(c, nu=1.5, lam=1e-3):
    return - (nu+1) * c / (nu*lam + c**2 + 1e-12)

# ---------- Training ----------
def train_one_epoch(model, wave, loader, device, args, optim, epoch, writer):
    """
    Stable training epoch for PWFEM:
    - Adaptive SVGD kernel bandwidth
    - Momentum and diffusion (SVLD)
    - Gradient normalization and clipping
    - Student-t prior regularization
    """
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    svgd_step = args.svgd_step * (0.9 ** (epoch / 1))
    prior_weight = args.prior_weight
    momentum = 0.9
    noise_scale = 1e-6

    for sample in pbar:
        u_lr = sample['u_lr'][0].numpy()
        up = zoom(u_lr, zoom=args.upsample, order=3)
        up_t = torch.tensor(up[None, None, :, :], dtype=torch.float32, device=device)

        # Wavelet decomposition
        Yl, Yh = wave.dwt(up_t)
        base = Yl.detach()
        M = args.particles

        # Initialize particles & velocity
        particles = base.repeat(M, 1, 1, 1).clone().detach().to(device).requires_grad_(True)
        velocity = torch.zeros_like(particles, device=device)

        # Define log-posterior
        def logpost_func(p_batch):
            recon = wave.idwt((p_batch, [h.repeat(p_batch.size(0), 1, 1, 1, 1) for h in Yh]))
            recon = (recon - recon.mean()) / (recon.std() + 1e-8)
            pred = model(recon)
            target = torch.tensor(u_lr, dtype=torch.float32, device=device)[None, None, :, :].repeat(p_batch.size(0), 1, 1, 1)
            loglik = -0.5 * ((pred - target) ** 2).mean() / (args.eps_lik ** 2)
            logprior = -0.5 * (args.nu + 1) * torch.log1p((p_batch ** 2) / (args.nu * args.lam)).mean()
            return loglik + prior_weight * logprior

        # SVGD + Momentum iterations
        for _ in range(args.svgd_steps_train):
            particles, velocity = svgd_step(
                particles, velocity, logpost_func,
                step_size=svgd_step,
                momentum=momentum,
                noise_scale=noise_scale
            )

        # Reconstruction and model training
        recon = wave.idwt((particles, [h.repeat(M, 1, 1, 1, 1) for h in Yh]))
        pred = model(recon)
        target = torch.tensor(u_lr, dtype=torch.float32, device=device)[None, None, :, :].repeat(M, 1, 1, 1)

        loss = ((pred - target) ** 2).mean() / (args.eps_lik ** 2)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    avg_loss = total_loss / len(loader)
    writer.add_scalar("train/loss", avg_loss, epoch)
    return avg_loss

@torch.no_grad()
def validate(model, wave, loader, device, args, epoch, writer):
    model.eval()
    mse_list = []
    for batch in loader:
        u_lr = batch['u_lr'][0].numpy()
        u_hr = batch['u_hr'][0].numpy()
        # bicubic
        bic = zoom(u_lr, zoom=args.upsample, order=3)
        mse_list.append(mean_squared_error(u_hr.reshape(-1), bic.reshape(-1)))
    writer.add_scalar("val/mse_bic", np.mean(mse_list), epoch)
    return np.mean(mse_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--particles", type=int, default=20)
    parser.add_argument("--svgd_steps", type=int, default=20)
    parser.add_argument("--svgd_step", type=float, default=1e-2)
    parser.add_argument("--eps_lik", type=float, default=0.1)
    parser.add_argument("--nu", type=float, default=1.5)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--upsample", type=int, default=4)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    train_ds = NPZDataset(args.train_dir)
    val_ds = NPZDataset(args.val_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    wave = WaveletModule().to(device)
    model = DownscaleResNet().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    for ep in range(1, args.epochs+1):
        loss = train_one_epoch(model, wave, train_loader, device, args, optim, ep, writer)
        mse_val = validate(model, wave, val_loader, device, args, ep, writer)
        torch.save({"model": model.state_dict()}, os.path.join(args.checkpoint_dir, f"epoch{ep}.pth"))
        print(f"[Epoch {ep}] train loss={loss:.4f}, val bic MSE={mse_val:.6f}")

    writer.close()

if __name__ == "__main__":
    main()
