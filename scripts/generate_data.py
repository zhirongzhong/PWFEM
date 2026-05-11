from typing import *
#!/usr/bin/env python3
"""
Usage:
  python scripts/generate_data.py --out_dir data/poisson/train --n 1000 --lr_side 40 --upsample 4 --seed 42
  python scripts/generate_data.py --out_dir data/poisson/test --n 100 --lr_side 40 --upsample 4 --seed 3407
"""

import os, argparse, time, random
import numpy as np
from scipy.ndimage import zoom
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from tqdm import trange

def build_poisson_A(n):
    N = n*n
    main = 4.0*np.ones(N)
    off1 = -1.0*np.ones(N-1)
    offn = -1.0*np.ones(N-n)
    A = diags([main,off1,off1,offn,offn],[0,-1,1,-n,n],shape=(N,N)).tocsr()
    for i in range(1,n):
        A[i*n, i*n-1] = 0.0
        A[i*n-1, i*n] = 0.0
    return A

def solve_poisson(n, f_func, d_boundary=0.0):
    h = 1.0/(n+1)
    xs = np.linspace(h,1-h,n)
    ys = np.linspace(h,1-h,n)
    X,Y = np.meshgrid(xs,ys,indexing='xy')
    f = f_func(X,Y).reshape(-1)
    A = build_poisson_A(n)
    b = (f*h*h).astype(float)
    if d_boundary != 0.0:
        for i in range(n):
            b[i] += d_boundary
            b[(n-1)*n + i] += d_boundary
            b[i*n] += d_boundary
            b[i*n + (n-1)] += d_boundary
    u = spsolve(A,b)
    return u.reshape((n,n)).astype(np.float32)

def forcing_factory(a,b,c):
    def f(x,y):
        return a*np.sin(b*x)*np.cos(c*y) + b*np.cos(a*x)*np.sin(c*y)
    return f

def sample_forcing_params():
    a = random.uniform(-4,4)
    b = random.uniform(-3,3)
    c = random.uniform(0,3)
    d = random.uniform(-2,2)
    return a,b,c,d

def generate_and_save(out_dir, idx, lr_side, upsample, seed=None):
    if seed is not None:
        random.seed(seed + idx)
        np.random.seed(seed + idx)
    a,b,c,d = sample_forcing_params()
    f = forcing_factory(a,b,c)
    hr_side = lr_side * upsample
    u_hr = solve_poisson(hr_side, f, d_boundary=d)
    u_lr = zoom(u_hr, zoom=1.0/upsample, order=3).astype(np.float32)
    fname = os.path.join(out_dir, f"sample_{idx:06d}.npz")
    np.savez_compressed(fname,
                        u_hr=u_hr.astype(np.float32),
                        u_lr=u_lr.astype(np.float32),
                        params=np.array([a,b,c,d],dtype=np.float32))
    return fname

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    start = time.time()
    for i in trange(args.n):
        generate_and_save(args.out_dir, i, args.lr_side, args.upsample, seed=args.seed)
    print("Saved {} samples to {}".format(args.n, args.out_dir))
    print("Elapsed {:.1f}s".format(time.time()-start))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True, default="data/train")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--lr_side", type=int, default=40)
    parser.add_argument("--upsample", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
