from typing import *
#!/usr/bin/env python3
"""
BayesPWFEM testing (minimal modification)
✅ Replace zoom-based upsampling with trained BayesDL output as initial SR input
"""

import os
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import torch
from probsr.models import DownscaleResNet, WaveletModule
from probsr.svgd import svgd_step
from model_zoo.BayesDL import BayesDL
import time

# --------------------------
#  CONFIGURATION
# --------------------------
DATASETS = ["poisson", "diffusion", "shock", "frac", "crack"]
METHODS = ["PWFEM"]
DATA_DIR = "data"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# BayesDL checkpoint path pattern
BAYESDL_DIR = "checkpoints/bayesdl"

# --------------------------
#  FUNCTIONS
# --------------------------
def compute_metrics(pred, gt, num_bins=100, eps=1e-10):
    def compute_kld(x, y):
        x = x.flatten()
        y = y.flatten()
        min_v = min(x.min(), y.min())
        max_v = max(x.max(), y.max())
        p_hist, bin_edges = np.histogram(x, bins=num_bins, range=(min_v, max_v), density=True)
        q_hist, _ = np.histogram(y, bins=num_bins, range=(min_v, max_v), density=True)
        p = p_hist / (np.sum(p_hist) + eps)
        q = q_hist / (np.sum(q_hist) + eps)
        return np.sum(p * np.log((p + eps) / (q + eps)))

    if pred.ndim == 3:
        mse, psnr, ssim_v, kld = [], [], [], []
        for c in range(pred.shape[0]):
            m = np.mean((pred[c] - gt[c]) ** 2)
            p_val = 10 * np.log10(1.0 / (m + 1e-12))
            s = ssim(pred[c], gt[c], data_range=1.0)
            k = compute_kld(gt[c], pred[c])
            mse.append(m)
            psnr.append(p_val)
            ssim_v.append(s)
            kld.append(k)
        return np.mean(mse), np.mean(psnr), np.mean(ssim_v), np.mean(kld)
    else:
        mse = np.mean((pred - gt) ** 2)
        psnr = 10 * np.log10(1.0 / (mse + 1e-12))
        ssim_v = ssim(pred, gt, data_range=1.0)
        kld = compute_kld(gt, pred)
        return mse, psnr, ssim_v, kld


# --------------------------
#  PWFEM Inference (modified)
# --------------------------
def infer_pwpfem(model, wave, bayesdl, u_lr, args, device, sample_id=None, save_path=None):
    """Perform inference with PWFEM, using BayesDL output instead of zoom"""
    # Prepare LR tensor
    if u_lr.ndim == 2:
        inp = torch.tensor(u_lr[None, None, :, :], dtype=torch.float32, device=device)
    elif u_lr.ndim == 3:
        inp = torch.tensor(u_lr[None, :, :, :], dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unexpected u_lr shape: {u_lr.shape}")

    if bayesdl is not None:
        bayesdl.eval()
        with torch.no_grad():
            out = bayesdl(inp)
            up_t = out[0] if isinstance(out, (list, tuple)) else out
        # print("[Info] BayesDL SR used as initial input.")
    else:
        up_t = torch.nn.functional.interpolate(inp, scale_factor=args['upsample'], mode="bicubic", align_corners=False)
        print("[Warning] BayesDL not found, using bicubic upsampling.")

    Yl, Yh = wave.dwt(up_t)

    M = args['particles']
    particles = Yl.repeat(M, 1, 1, 1).clone().detach().requires_grad_(True)
    velocity = torch.zeros_like(particles, device=device)

    particle_history = []
    grad_history = []

    nu = torch.tensor(args['nu'], dtype=torch.float32, device=device)
    lam = torch.tensor(args['lam'], dtype=torch.float32, device=device)

    def logpost_func(p_batch):
        recon = wave.idwt((p_batch, [h.repeat(p_batch.size(0), 1, 1, 1, 1) for h in Yh]))
        recon = (recon - recon.mean()) / (recon.std() + 1e-8)
        pred = model(recon)
        if u_lr.ndim == 2:
            target = torch.tensor(u_lr, dtype=torch.float32, device=device)[None, None, :, :].repeat(p_batch.size(0), 1, 1, 1)
        else:
            target = torch.tensor(u_lr, dtype=torch.float32, device=device)[None, :, :, :].repeat(p_batch.size(0), 1, 1, 1)
        loglik = -0.5 * ((pred - target) ** 2).mean() / (args['eps_lik'] ** 2)
        logprior = -0.5 * (nu + 1) * torch.log1p((p_batch ** 2) / (nu * lam + 1e-8)).mean()
        return loglik + args['prior_weight'] * logprior

    for _ in range(args['svgd_steps_infer']):
        if sample_id:
            with torch.no_grad():
                recon_step = wave.idwt((particles.detach(), [h.repeat(M, 1, 1, 1, 1) for h in Yh]))
            particle_history.append(recon_step.cpu().numpy().astype(np.float32))
        particles, velocity = svgd_step(particles, velocity, logpost_func,
                                        step_size=args['svgd_step'], momentum=0.9, noise_scale=1e-3)
        if sample_id:
            grad_history.append(velocity.detach().cpu().numpy().astype(np.float32))

    if sample_id and save_path:
        np.savez_compressed(os.path.join(save_path, f"{sample_id}_particles"), np.array(particle_history))
        np.savez_compressed(os.path.join(save_path, f"{sample_id}_grads"), np.array(grad_history))
    recon_particles = wave.idwt((particles, [h.repeat(M,1,1,1,1) for h in Yh]))
    recon_np = recon_particles.detach().cpu().numpy()
    uq_map = np.std(recon_np.squeeze(), axis=0, ddof=1)  # posterior std
    return recon_np.mean(0).squeeze(), uq_map

# --------------------------
#  Run case study
# --------------------------
def run_case(dataset, method, pwpfem_ckpt=None):
    npz_dir = os.path.join(DATA_DIR, dataset, "test")
    npz_files = sorted([f for f in os.listdir(npz_dir) if f.endswith(".npz")])
    metrics = []
    inference_times = []

    OUT_DIR = "results"
    hist_save_path = os.path.join(OUT_DIR, "svgd_history", dataset)
    os.makedirs(hist_save_path, exist_ok=True)

    print(f"\nRunning dataset: {dataset}, method: {method}  ({len(npz_files)} samples)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wave = WaveletModule().to(device)

    # Auto-detect channels
    dummy = np.load(os.path.join(npz_dir, npz_files[0]))["u_lr"].astype(np.float32)
    in_ch = 1 if dummy.ndim == 2 else dummy.shape[0]
    model = DownscaleResNet(in_ch=in_ch, out_size=(40, 40)).to(device)

    ckpt = torch.load(pwpfem_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    nu_val = ckpt.get("nu", torch.tensor(1.5)).item()
    lam_val = ckpt.get("lam", torch.tensor(1e-3)).item()
    print(f"Loaded Student-t prior: nu={nu_val:.4f}, lam={lam_val:.6e}")

    pwpfem_args = dict(
        upsample=4,
        particles=8,
        svgd_steps_infer=200,
        svgd_step=1e-5,
        eps_lik=0.3,
        nu=nu_val,
        lam=lam_val,
        prior_weight=1e-4
    )


    # Load BayesDL model
    bayesdl_ckpt = os.path.join(BAYESDL_DIR, f"BayesDL_{dataset}_best.pth")
    if os.path.exists(bayesdl_ckpt):
        bayesdl = BayesDL(
            in_channels=in_ch, out_channels=in_ch, sr_scale=4,
            learn_std=True, likelihood="student",
            drop=0.05, drop2d=0.05,
            n_resgroups=5, n_resblocks=10, n_feat=64
        ).to(device)
        bayesdl.load_state_dict(torch.load(bayesdl_ckpt, map_location=device), strict=False)
        print(f"✅ Loaded BayesDL init from {bayesdl_ckpt}")
    else:
        print(f"⚠️ BayesDL checkpoint not found for {dataset}, fallback to bicubic.")
        bayesdl = None

    for f in tqdm(npz_files, desc=f"{dataset}-{method}"):

        data = np.load(os.path.join(npz_dir, f))
        u_hr = data["u_hr"].astype(np.float32)
        u_lr = data["u_lr"].astype(np.float32)

        # normalize
        u_hr = (u_hr - u_hr.min()) / (u_hr.max() - u_hr.min() + 1e-12)
        u_lr = (u_lr - u_lr.min()) / (u_lr.max() - u_lr.min() + 1e-12)

        start_time = time.perf_counter()

        if method == "PWFEM":
            sample_id = f.replace(".npz", "")
            recon_mean, recon_std = infer_pwpfem(model, wave, bayesdl, u_lr, pwpfem_args, device, sample_id=sample_id, save_path=hist_save_path)
            recon, uq_map = recon_mean, recon_std
        else:
            raise NotImplementedError("Only PWFEM supported here.")

        # crop to match HR
        if recon.ndim == 3:
            _, H, W = u_hr.shape
            recon = recon[:, :H, :W]
            uq_map = uq_map[:, :H, :W]
        else:
            H, W = u_hr.shape
            recon = recon[:H, :W]
            uq_map = uq_map[:H, :W]


        end_time = time.perf_counter()
        inference_times.append(end_time - start_time)

        mse, psnr, ssim_val, kld = compute_metrics(recon, u_hr)
        squared_error = np.sqrt((u_hr - recon) ** 2)
        calib = np.abs(np.mean((squared_error / (np.abs(uq_map) + 1e-12)) - 1))
        metrics.append([mse, psnr, ssim_val, calib, kld])

        np.savez_compressed(
            os.path.join(OUT_DIR, f"result_{dataset}_{method}_{f}"),
            u_hr=u_hr, u_lr=u_lr, recon=recon, uq=uq_map
        )
        del u_hr, recon, uq_map

    metrics = np.array(metrics)
    np.savez_compressed(os.path.join(OUT_DIR, f"metrics_{dataset}_{method}.npz"), metrics=metrics)
    print(f"Saved metrics for {dataset} [{method}], mean PSNR={metrics[:,1].mean():.2f}")

    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    print(f"Avg. Inference time / sample: {mean_time:.3f} ± {std_time:.3f} s")

# --------------------------
#  MAIN
# --------------------------
if __name__ == "__main__":
    DATASETS = ["poisson"]
    # DATASETS = ["poisson", "diffusion", "shock", "frac", "crack"]
    for ds in DATASETS:
        PWFEM_CHECKPOINT = f"checkpoints/bayespwfem/{ds}/epoch_best.pth"
        for method in METHODS:
            run_case(ds, method, pwpfem_ckpt=PWFEM_CHECKPOINT)
