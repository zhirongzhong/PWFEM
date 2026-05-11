from typing import *
#!/usr/bin/env python3
"""
Inference script (example)
Usage:
  python scripts/infer.py --checkpoint checkpoints/model.pth --test_dir data/test --out_dir outputs --M 50 --steps 200
"""
import argparse, os
import numpy as np
import torch
from probsr.dataset import NPZDataset
from probsr.models import WaveletModule, DownscaleResNet
from probsr.svgd import svgd_step
from probsr.utils import save_grid_imgs
from scipy.ndimage import zoom

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--M", type=int, default=50)
    p.add_argument("--steps", type=int, default=200)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model + wave
    wave = WaveletModule().to(device)
    model = DownscaleResNet().to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck['model'])
    # dataset
    ds = NPZDataset(args.test_dir)
    for i in range(len(ds)):
        sample = ds[i]
        u_lr = sample['u_lr']
        u_hr = sample['u_hr']
        # init
        up = zoom(u_lr, zoom=int(u_hr.shape[0]/u_lr.shape[0]), order=3)
        up_t = torch.tensor(up[None,None,:,:], dtype=torch.float32, device=device)
        Yl, Yh = wave.dwt(up_t)
        M = args.particles
        particles = Yl.repeat(M, 1, 1, 1).clone().detach().requires_grad_(True)
        velocity = torch.zeros_like(particles, device=device)
        # TODO: build particles from coeffs and run SVGD (similar to train)
        # For brevity, here we simulate: use bicubic as baseline and save
        bic = up
        save_grid_imgs(args.out_dir, f"sample_{i:04d}", u_lr, u_hr, bic, bic, rec_std=np.zeros_like(bic))
        if i>=20: break

if __name__ == "__main__":
    main()
