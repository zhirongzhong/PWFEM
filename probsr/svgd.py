from typing import *
"""
"""

import torch

def rbf_kernel_matrix(X):
    """RBF kernel for SVGD"""
    pairwise = torch.cdist(X, X)**2
    med = torch.median(pairwise.detach())
    h = med / torch.log(torch.tensor(X.size(0), dtype=torch.float32, device=X.device)+1.0)
    h = torch.clamp(h, min=1e-6)
    K = torch.exp(-pairwise/h)
    return K, h

def svgd_step(particles, velocity, logpost_func, step_size=1e-6, momentum=0.9, noise_scale=1e-6):
    """
    Kernelized SVGD update with momentum and diffusion.
    Implements adaptive RBF kernel and repulsive force correction.
    """
    M = particles.size(0)
    particles_det = particles.detach().clone().requires_grad_(True)

    logp = logpost_func(particles_det)
    grads = torch.autograd.grad(logp, particles_det, retain_graph=False)[0]
    grads = torch.nan_to_num(grads, nan=0.0)

    flat = particles_det.view(M, -1)
    with torch.no_grad():
        pdists = torch.cdist(flat, flat, p=2)
        med_sq = torch.median(pdists) ** 2 + 1e-8
        h = med_sq / torch.log(torch.tensor(M + 1.0, device=particles.device))
        kernel = torch.exp(-pdists ** 2 / h)
        grad_kernel_flat = -2 * torch.matmul(kernel, flat) / h
        grad_kernel = grad_kernel_flat.view_as(particles_det)

    phi = (torch.matmul(kernel, grads.view(M, -1)).view_as(particles_det) + grad_kernel) / float(M)
    phi = torch.nan_to_num(phi, nan=0.0)
    phi = phi / (torch.norm(phi) + 1e-8)  # normalize for stability

    velocity = momentum * velocity + phi
    particles_new = particles + step_size * velocity + noise_scale * torch.randn_like(particles)
    particles_new = particles_new.detach().clone().requires_grad_(True)

    return particles_new, velocity