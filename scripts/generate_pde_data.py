from typing import *
"""
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags, lil_matrix
from scipy.ndimage import gaussian_filter


# ---------------- Poisson ----------------
def _build_poisson_A(n):
    """5-point Laplacian matrix with Dirichlet boundary conditions."""
    N = n * n
    main = 4.0 * np.ones(N)
    off1 = -1.0 * np.ones(N - 1)
    offn = -1.0 * np.ones(N - n)
    A = diags([main, off1, off1, offn, offn], [0, -1, 1, -n, n], shape=(N, N)).tocsr()
    for i in range(1, n):
        A[i * n, i * n - 1] = 0.0
        A[i * n - 1, i * n] = 0.0
    return A

def _forcing_factory(a, b, c):
    return lambda x, y: a * np.sin(b * np.pi * x) * np.cos(c * np.pi * y) + \
                        b * np.cos(a * np.pi * x) * np.sin(c * np.pi * y)
# ---------------- Poisson ----------------
def generate_poisson(hr_side, lr_side, rng=None):
    if rng is None: rng = np.random
    a, b, c, d = rng.uniform(-4, 4), rng.uniform(-3, 3), rng.uniform(0, 3), rng.uniform(-2, 2)

    results = []
    for n in [hr_side, lr_side]:
        h = 1.0 / (n + 1)
        X, Y = np.meshgrid(np.linspace(h, 1 - h, n), np.linspace(h, 1 - h, n), indexing='xy')
        f = _forcing_factory(a, b, c)(X, Y)
        A = _build_poisson_A(n)
        b_vec = (f * h * h).reshape(-1)
        if abs(d) > 1e-6:
            for i in range(n):
                b_vec[i] += d;
                b_vec[(n - 1) * n + i] += d
                b_vec[i * n] += d;
                b_vec[i * n + (n - 1)] += d
        u = spsolve(A, b_vec).reshape(n, n).astype(np.float32)
        results.append(u)
    return results[0], results[1]  # u_hr, u_lr


# ---------------- Multiscale diffusion ----------------
def generate_multiscale_diffusion(hr_side, lr_side, rng=None, contrast=50):
    if rng is None: rng = np.random.default_rng()
    a, b, c = rng.uniform(0.5, 1, 3)
    n_blocks = rng.randint(5, 15)
    blocks = [(rng.random(), rng.random(), rng.uniform(0.05, 0.2), rng.uniform(0.05, 0.2),
               contrast if rng.random() > 0.5 else 1.0) for _ in range(n_blocks)]

    results = []
    for N in [hr_side, lr_side]:
        h = 1.0 / (N + 1)
        coeff = np.ones((N, N), dtype=np.float32)
        for (rx, ry, rw, rh, val) in blocks:
            x0, y0 = int(rx * (N - 1)), int(ry * (N - 1))
            bw, bh = max(1, int(rw * N)), max(1, int(rh * N))
            coeff[y0:y0 + bh, x0:x0 + bw] = val

        X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
        f = _forcing_factory(a, b, c)(X, Y) + 0.1 * rng.standard_normal((N, N))
        A = lil_matrix((N * N, N * N))
        b_vec = (f * h ** 2).reshape(-1)
        for i in range(N):
            for j in range(N):
                idx = i * N + j
                if i in [0, N - 1] or j in [0, N - 1]:
                    A[idx, idx] = 1.0;
                    b_vec[idx] = 0.0;
                    continue
                aE, aW = 0.5 * (coeff[i, j] + coeff[i, j + 1]), 0.5 * (coeff[i, j] + coeff[i, j - 1])
                aN, aS = 0.5 * (coeff[i, j] + coeff[i - 1, j]), 0.5 * (coeff[i, j] + coeff[i + 1, j])
                A[idx, idx] = aE + aW + aN + aS
                A[idx, idx + 1], A[idx, idx - 1], A[idx, idx + N], A[idx, idx - N] = -aE, -aW, -aS, -aN
        u = spsolve(A.tocsr(), b_vec).reshape(N, N).astype(np.float32)
        results.append(u)
    return results[0], results[1]


# ---------------- 2D Euler Shock ----------------
def generate_shocktube(hr_side=160, lr_side=40, t_end=None, rng=None):
    if rng is None: rng = np.random.default_rng()
    states = {'rL': rng.uniform(0.8, 1.4), 'rR': rng.uniform(0.1, 0.7),
              'pL': rng.uniform(0.8, 2.5), 'pR': rng.uniform(0.05, 0.4),
              'ang': rng.uniform(0, np.pi / 2), 'x0': rng.uniform(0.3, 0.7)}
    t = t_end if t_end else rng.uniform(0.05, 0.2)
    nx, ny = np.cos(states['ang']), np.sin(states['ang'])
    s_shock = (np.sqrt(1.4 * states['pL'] / states['rL']) + np.sqrt(1.4 * states['pR'] / states['rR'])) / 2 + 0.1

    results = []
    for N in [hr_side, lr_side]:
        X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N), indexing='xy')
        dist = (X - states['x0']) * nx + (Y - states['x0']) * ny
        shift = s_shock * t
        rho = np.where(dist < -shift, states['rL'],
                       np.where(dist > shift, states['rR'], 0.5 * (states['rL'] + states['rR'])))
        rho += 0.02 * rng.standard_normal((N, N))
        results.append(np.clip(rho, 0, None).astype(np.float32))
    return results[0], results[1]


# ---------------- Fractional Laplacian ----------------
def generate_frac_laplacian(hr_side, lr_side, alpha=1.5, rng=None):
    if rng is None: rng = np.random
    noise_fft_hr = rng.randn(hr_side, hr_side) + 1j * rng.randn(hr_side, hr_side)

    results = []
    for N in [hr_side, lr_side]:
        if N == hr_side:
            n_fft = noise_fft_hr
        else:
            n_fft = np.zeros((N, N), dtype=complex)
            m = N // 2
            n_fft[:m, :m] = noise_fft_hr[:m, :m]
            n_fft[-m:, :m] = noise_fft_hr[-m:, :m]
            n_fft[:m, -m:] = noise_fft_hr[:m, -m:]
            n_fft[-m:, -m:] = noise_fft_hr[-m:, -m:]

        kx, ky = np.fft.fftfreq(N)[:, None], np.fft.fftfreq(N)[None, :]
        spectrum = (1.0 / (kx ** 2 + ky ** 2 + 1e-6)) ** (alpha / 2.0)
        u = np.fft.ifft2(n_fft * spectrum).real
        results.append(((u - u.mean()) / (u.std() + 1e-12)).astype(np.float32))
    return results[0], results[1]



######################################################################
def _solve_displacement_pde(N, d, load):
    """物理自洽的位移场求解器"""
    h = 1.0 / (N - 1)
    num_nodes = N * N
    kappa = (1.0 - d) ** 2 + 1e-7

    A = lil_matrix((num_nodes, num_nodes))
    b = np.zeros(num_nodes)

    for i in range(N):
        for j in range(N):
            idx = i * N + j
            if i == 0:
                A[idx, idx] = 1.0
                b[idx] = 0.0
            elif i == N - 1:
                A[idx, idx] = 1.0
                b[idx] = load
            else:
                kn = 0.5 * (kappa[i, j] + kappa[i + 1, j])
                ks = 0.5 * (kappa[i, j] + kappa[i - 1, j])
                ke = 0.5 * (kappa[i, j] + kappa[i, min(j + 1, N - 1)])
                kw = 0.5 * (kappa[i, j] + kappa[i, max(j - 1, 0)])
                A[idx, idx] = (kn + ks + ke + kw)
                A[idx, idx - N], A[idx, idx + N] = -ks, -kn
                A[idx, idx - 1], A[idx, idx + 1] = -kw, -ke

    return spsolve(A.tocsr(), b).reshape(N, N).astype(np.float32)


def _solve_phase_implicit(N, d_old, H, ell, Gc):
    h = 1.0 / (N - 1)
    num_nodes = N * N
    A = lil_matrix((num_nodes, num_nodes))
    rhs = (2.0 * H).flatten()

    c_lap = Gc * ell / (h ** 2)
    c_react = Gc / ell

    for i in range(N):
        for j in range(N):
            idx = i * N + j
            if d_old[i, j] > 0.9:
                A[idx, idx] = 1.0
                rhs[idx] = 1.0
                continue

            A[idx, idx] = 4.0 * c_lap + c_react + 2.0 * H[i, j]
            if i > 0:
                A[idx, idx - N] = -c_lap
            else:
                A[idx, idx] -= c_lap
            if i < N - 1:
                A[idx, idx + N] = -c_lap
            else:
                A[idx, idx] -= c_lap
            if j > 0:
                A[idx, idx - 1] = -c_lap
            else:
                A[idx, idx] -= c_lap
            if j < N - 1:
                A[idx, idx + 1] = -c_lap
            else:
                A[idx, idx] -= c_lap

    return spsolve(A.tocsr(), rhs).reshape(N, N).astype(np.float32)

def _solve_phase_field_consistent(N, ell, Gc, E, n_steps, load_max, nc, nw, nh):
    h = 1.0 / (N - 1)
    X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N), indexing="xy")

    d = np.zeros((N, N), dtype=np.float32)
    actual_nw = np.maximum(nw, h * 0.7)
    notch = ((X < nc[1] + nh) & (np.abs(Y - nc[0]) < actual_nw))
    d[notch] = 1.0

    H_max = np.zeros((N, N), dtype=np.float32)

    for step in range(n_steps):
        load = load_max * (step + 1) / n_steps
        u = _solve_displacement_pde(N, d, load)

        _, uy_y = np.gradient(u, h)
        H_curr = 0.5 * E * (uy_y ** 2)
        H_max = np.maximum(H_max, H_curr)

        rel_sigma = 0.5 * (N / 128.0)
        H_smooth = gaussian_filter(H_max, sigma=rel_sigma)

        d = _solve_phase_implicit(N, d, H_smooth, ell, Gc)
        d[notch] = 1.0

    u_final = _solve_displacement_pde(N, d, load_max)

    u_norm = (u_final - u_final.mean()) / (u_final.std() + 1e-8)
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)

    return np.stack([u_norm, d_norm], axis=0).astype(np.float32)


def generate_phase_field_fracture(hr_side, lr_side, rng=None, ell=0.05, Gc=1.0, E=20.0, n_steps=3, load_max=2.0):
    if rng is None: rng = np.random.RandomState()

    nc = (rng.uniform(0.3, 0.7), rng.uniform(0.1, 0.3))
    nw = rng.uniform(0.01, 0.015)
    nh = rng.uniform(0.02, 0.08)

    E_rand = E * rng.uniform(0.9, 1.1)
    Gc_rand = Gc * rng.uniform(0.9, 1.1)

    hr = _solve_phase_field_consistent(hr_side, ell, Gc_rand, E_rand, n_steps, load_max, nc, nw, nh)
    lr = _solve_phase_field_consistent(lr_side, ell, Gc_rand, E_rand, n_steps, load_max, nc, nw, nh)

    return hr, lr