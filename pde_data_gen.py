from typing import *
#!/usr/bin/env python3
"""
Unified high-precision PDE dataset generator (2025 edition)
===========================================================
Supports:
    - Poisson equation
    - Multiscale diffusion
    - 2D Euler shock (Riemann problem)
    - Fractional Laplacian random field
    - Allen–Cahn phase field model

Usage:
    python pde_data_gen.py --pde poisson
    python pde_data_gen.py --pde diffusion
    python pde_data_gen.py --pde shock
    python pde_data_gen.py --pde frac
    python pde_data_gen.py --pde allen
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from scripts.generate_pde_data import (
    generate_poisson,
    generate_multiscale_diffusion,
    generate_shocktube,
    generate_frac_laplacian,
    generate_phase_field_fracture
)

def save_npz(out_dir, idx, u_hr, u_lr):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"sample_{idx:05d}.npz")
    np.savez(path, u_hr=u_hr.astype(np.float32), u_lr=u_lr.astype(np.float32))


def generate_dataset(pde_type, out_dir, n_train=1000, n_test=200,
                     hr_side=160, lr_side=40, up=4, seed=42):
    """
    Generate train/test datasets. For reproducibility, each sample uses its own RNG
    initialized from the provided base seed + sample index.
    """
    base_out = os.path.join(out_dir, pde_type)
    train_dir = os.path.join(base_out, "train")
    test_dir = os.path.join(base_out, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    def sample_field(i, rng):
        if pde_type == "poisson":
            uh, ul = generate_poisson(hr_side, lr_side, rng=rng)
            return uh, ul
        elif pde_type == "diffusion":
            uh, ul = generate_multiscale_diffusion(hr_side, lr_side, rng=rng)
            return uh, ul
        elif pde_type == "shock":
            # 1D shock tiled to 2D
            uh, ul = generate_shocktube(hr_side, lr_side, rng=rng)
            return uh, ul
        elif pde_type == "frac":
            uh, ul = generate_frac_laplacian(hr_side, lr_side, alpha=1.5, rng=rng)
            return uh, ul
        elif pde_type == "crack":
            uh2d, ul2d = generate_phase_field_fracture(hr_side, lr_side, rng=rng)
            return uh2d, ul2d
        else:
            raise ValueError(f"Unknown PDE type: {pde_type}")

    print(f"Generating {pde_type} training set...")
    for i in tqdm(range(n_train)):
        rng = np.random.RandomState(seed + i)
        u_hr, u_lr = sample_field(i, rng)
        save_npz(train_dir, i, u_hr, u_lr)

    print(f"Generating {pde_type} test set...")
    for i in tqdm(range(n_test)):
        rng = np.random.RandomState(seed + 100000 + i)
        u_hr, u_lr = sample_field(i, rng)
        save_npz(test_dir, i, u_hr, u_lr)

    print(f"Done. Dataset saved in {base_out}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", required=True,
                        choices=["poisson", "diffusion", "shock", "frac", "crack"])
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--hr_side", type=int, default=160)
    parser.add_argument("--lr_side", type=int, default=40)
    parser.add_argument("--up", type=int, default=4)
    parser.add_argument("--train_n", type=int, default=1000)
    parser.add_argument("--test_n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.pde, args.out_dir,
                     n_train=args.train_n, n_test=args.test_n,
                     hr_side=args.hr_side, up=args.up, seed=args.seed)

if __name__ == "__main__":
    main()
