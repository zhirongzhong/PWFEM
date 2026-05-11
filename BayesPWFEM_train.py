from typing import *
#!/usr/bin/env python3
"""
BayesPWFEM training with BayesDL initialization
"""
import os, argparse
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from probsr.dataset import NPZDataset
from probsr.models import WaveletModule, DownscaleResNet
from probsr.svgd import svgd_step
from model_zoo.BayesDL import BayesDL
import time

# ==========================================================
#  Training with Trainable Student-t Prior
# ==========================================================
def train_epoch(model, wave, loader, device, args, optim, epoch, writer, nu_param, lam_param, bayesdl=None):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    svgd_steps = args.svgd_step * (0.9 ** (epoch / 5))
    prior_weight = args.prior_weight
    momentum = 0.9
    noise_scale = 1e-6
    epoch_times = []

    for sample in pbar:
        u_lr = sample['u_lr'][0].numpy()
        epoch_start_time = time.perf_counter()

        if bayesdl is not None:
            with torch.no_grad():
                if u_lr.ndim == 2:
                    inp = torch.tensor(u_lr[None, None, :, :], dtype=torch.float32, device=device)
                else:
                    inp = torch.tensor(u_lr[None, :, :, :], dtype=torch.float32, device=device)
                out = bayesdl(inp)
                up_t = out[0] if isinstance(out, (list, tuple)) else out
        else:
            if u_lr.ndim == 2:
                up = zoom(u_lr, zoom=args.upsample, order=3)
                up_t = torch.tensor(up[None, None, :, :], dtype=torch.float32, device=device)
            elif u_lr.ndim == 3:
                up = np.stack([zoom(ch, zoom=args.upsample, order=3) for ch in u_lr])
                up_t = torch.tensor(up[None, :, :, :], dtype=torch.float32, device=device)
            else:
                raise ValueError(f"Unexpected u_lr shape: {u_lr.shape}")

        # ==========================================================
        # Wavelet decomposition and SVGD inference
        # ==========================================================
        Yl, Yh = wave.dwt(up_t)
        base = Yl.detach()
        M = args.particles

        particles = base.repeat(M, 1, 1, 1).clone().detach().to(device).requires_grad_(True)
        velocity = torch.zeros_like(particles, device=device)

        def logpost_func(p_batch):
            recon = wave.idwt((p_batch, [h.repeat(p_batch.size(0), 1, 1, 1, 1) for h in Yh]))
            recon = (recon - recon.mean()) / (recon.std() + 1e-8)
            pred = model(recon)
            target = torch.tensor(u_lr, dtype=torch.float32, device=device)
            if u_lr.ndim == 2:
                target = target[None, None, :, :].repeat(p_batch.size(0), 1, 1, 1)
            else:
                target = target[None, :, :, :].repeat(p_batch.size(0), 1, 1, 1)

            loglik = -0.5 * ((pred - target) ** 2).mean() / (args.eps_lik ** 2)
            nu = torch.nn.functional.softplus(nu_param)
            lam = torch.nn.functional.softplus(lam_param)
            logprior = -0.5 * (nu + 1.0) * torch.mean(torch.log1p((p_batch ** 2) / (nu * lam + 1e-8)))
            return loglik + prior_weight * logprior

        for _ in range(args.svgd_steps_train):
            particles, velocity = svgd_step(
                particles, velocity, logpost_func,
                step_size=svgd_steps,
                momentum=momentum,
                noise_scale=noise_scale
            )

        recon = wave.idwt((particles, [h.repeat(M, 1, 1, 1, 1) for h in Yh]))
        pred = model(recon)
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        if u_lr.ndim == 2:
            target = torch.tensor(u_lr, dtype=torch.float32, device=device)[None, None, :, :].repeat(M, 1, 1, 1)
        else:
            target = torch.tensor(u_lr, dtype=torch.float32, device=device)[None, :, :, :].repeat(M, 1, 1, 1)

        mse_loss = ((pred - target) ** 2).mean() / (args.eps_lik ** 2)
        nu = torch.nn.functional.softplus(nu_param)
        lam = torch.nn.functional.softplus(lam_param)
        prior_term = 0.5 * (nu + 1.0) * torch.mean(torch.log1p((particles ** 2) / (nu * lam + 1e-8)))
        loss = mse_loss + prior_weight * prior_term

        optim.zero_grad()
        loss.backward()
        params_to_clip = [p for p in model.parameters()] + [nu_param, lam_param]
        torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
        optim.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    avg_loss = total_loss / len(loader)
    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/nu_value", torch.nn.functional.softplus(nu_param).item(), epoch)
    writer.add_scalar("train/lam_value", torch.nn.functional.softplus(lam_param).item(), epoch)
    if epoch % 100 == 0:
        nu_val = torch.nn.functional.softplus(nu_param).detach().cpu()
        lam_val = torch.nn.functional.softplus(lam_param).detach().cpu()
        print(f"[Epoch {epoch}] nu={nu_val.item():.4f}, lam={lam_val.item():.6e}, loss={avg_loss:.6f}")
        torch.save({
            "model": model.state_dict(),
            "nu": nu_val,
            "lam": lam_val
        }, os.path.join(args.checkpoint_dir, f"epoch{epoch}.pth"))

    mean_time = np.mean(epoch_times)
    std_time = np.std(epoch_times)
    print(f"Avg. Train time / epoch: {mean_time:.3f} ± {std_time:.3f} s")

    return avg_loss


# ==========================================================
#  Main
# ==========================================================
def main(ds="poisson"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default=f"data/{ds}/train")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--svgd_steps_train", type=int, default=20)
    parser.add_argument("--svgd_step", type=float, default=1e-4)
    parser.add_argument("--eps_lik", type=float, default=0.3)
    parser.add_argument("--nu", type=float, default=1.5)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--upsample", type=int, default=4)
    parser.add_argument("--checkpoint_dir", default=f"checkpoints/bayespwfem/{ds}")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--prior_weight", type=float, default=1e-2)
    parser.add_argument("--bayesdl_ckpt", type=str, default=f"checkpoints/bayesdl/BayesDL_{ds}_final.pth")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    train_ds = NPZDataset(args.train_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    sample = next(iter(train_loader))["u_lr"][0].numpy()
    in_channels = 1 if sample.ndim == 2 else sample.shape[0]

    wave = WaveletModule().to(device)
    model = DownscaleResNet(in_ch=in_channels, out_size=(40, 40)).to(device)

    nu_param = torch.nn.Parameter(torch.tensor(args.nu, device=device))
    lam_param = torch.nn.Parameter(torch.tensor(args.lam, device=device))

    optim = torch.optim.Adam([
        {"params": model.parameters(), "lr": 1e-3},
        {"params": [nu_param, lam_param], "lr": 1e-4}
    ])

    bayesdl = None
    if os.path.exists(args.bayesdl_ckpt):
        print(f"✅ Loading BayesDL from {args.bayesdl_ckpt}")
        bayesdl = BayesDL(in_channels=in_channels,
                          out_channels=in_channels,
                          sr_scale=args.upsample,
                          learn_std=True,
                          likelihood="student",
                          drop=0.05,
                          drop2d=0.05,
                          n_resgroups=5,
                          n_resblocks=10,
                          n_feat=64).to(device)
        state = torch.load(args.bayesdl_ckpt, map_location=device)
        bayesdl.load_state_dict(state, strict=False)
        bayesdl.eval()
    else:
        print("⚠️ BayesDL checkpoint not found, using bicubic upsampling fallback.")

    for ep in range(1, args.epochs + 1):
        train_epoch(model, wave, train_loader, device, args, optim, ep, writer, nu_param, lam_param, bayesdl)

    writer.close()
    print("Training completed.")


if __name__ == "__main__":
    # DATASETS = ["poisson", "diffusion", "shock", "frac", "crack"]
    DATASETS = ["poisson"]
    for ds in DATASETS:
        main(ds)
