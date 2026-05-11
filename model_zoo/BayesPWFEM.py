from typing import *
"""
model_zoo/BayesPWFEM.py: module for the Bayesian PWFEM / ProbSR project.
Auto-documented for open-source release.
"""

# model_zoo/BayesPWFEM.py
# Minimal BayesPWFEM model with trainable Student-t prior params nu and lam.
import torch
import torch.nn as nn
import torch.nn.functional as F
from probsr.svgd import svgd_step
from probsr.models2 import WaveletModule, DownscaleResNet
import numpy as np

class BayesPWFEM(nn.Module):
    def __init__(self,
                 in_ch=1,
                 sr_scale=4,
                 particles=32,
                 svgd_steps=30,
                 svgd_step_size=1e-4,
                 prior_weight=1e-4,
                 eps_lik=0.3,
                 device=None,
                 trainable_prior=True,
                 init_nu=1.5,
                 init_lam=1e-3):
        """
        Minimal wrapper: performs SVGD on wavelet lowpass coefficients and reconstructs HR via IDWT + DownscaleResNet.
        We add trainable Student-t prior params (nu, lam).
        """
        super().__init__()
        self.in_ch = in_ch
        self.sr_scale = sr_scale
        self.particles = particles
        self.svgd_steps = svgd_steps
        self.svgd_step_size = svgd_step_size
        self.prior_weight = prior_weight
        self.eps_lik = eps_lik
        self.device = device if device is not None else torch.device("cpu")

        # wavelet module & SR network (DownscaleResNet returns HR recon when given reconstructed full image)
        self.wavelet = WaveletModule().to(self.device)
        self.net = DownscaleResNet(in_ch=self.in_ch, out_size=(40,40)).to(self.device)

        # trainable prior params
        self.trainable_prior = trainable_prior
        if self.trainable_prior:
            # store unconstrained params as trainable; apply softplus in use to enforce positivity
            self.nu_param = nn.Parameter(torch.tensor(float(init_nu), device=self.device))
            self.lam_param = nn.Parameter(torch.tensor(float(init_lam), device=self.device))
        else:
            self.register_buffer('nu_val', torch.tensor(float(init_nu)))
            self.register_buffer('lam_val', torch.tensor(float(init_lam)))
            self.nu_param = None
            self.lam_param = None

    def get_nu_lam(self):
        if self.trainable_prior:
            # softplus to ensure positive
            nu = F.softplus(self.nu_param) + 1e-6
            lam = F.softplus(self.lam_param) + 1e-12
        else:
            nu = self.nu_val
            lam = self.lam_val
        return nu, lam

    def forward(self, u_lr, use_train_prior=True, override_nu_lam=None):
        """
        Forward runs SVGD inference for self.particles, reconstructs HR (mean over particles),
        then passes reconstructed images through self.net to produce final HR recon.
        - u_lr: tensor shape (B, C, H, W) -> we assume B typically 1
        - override_nu_lam: optional tuple (nu, lam) to use explicit values
        Returns: reconstructed HR mean (B, C, H*scale, W*scale)
        """
        B = u_lr.shape[0]
        # upsample LR to high-res grid (bicubic)
        up = F.interpolate(u_lr, scale_factor=self.sr_scale, mode='bilinear', align_corners=False)

        # wavelet lowpass & highpass
        Yl, Yh = self.wavelet.dwt(up)  # Yl: (B, C, H', W'), Yh: list of detail tensors

        M = self.particles
        # initialize particles by repeating Yl M times
        particles = Yl.detach().repeat(M, 1, 1, 1, 1).to(self.device)  # (M*B?, but we assume B==1)
        # To support batch >1, we will reshape: we assume B==1 for simplicity in minimal change
        velocity = torch.zeros_like(particles, device=self.device)

        # obtain nu/lam
        if override_nu_lam is not None:
            nu, lam = override_nu_lam
        else:
            nu, lam = self.get_nu_lam()

        # define logpost for SVGD; SVGD implementation will call it with particles batched
        def logpost_func(p_batch):
            # p_batch shape: (M, C, h, w)
            # expand Yh to match batch (implementation depends on WaveletModule internals)
            # reconstruct images from wavelet (careful: wavelet.idwt expects tuple (Yl, Yh))
            # here we assume we can call idwt with (p_batch, Yh_expanded)
            # expand Yh for each particle (minimal expand strategy)
            Yh_exp = [h.repeat(p_batch.size(0), *([1] * (h.dim() - 1))) for h in Yh]
            recon = self.wavelet.idwt((p_batch, Yh_exp))
            recon = (recon - recon.mean()) / (recon.std() + 1e-8)
            pred = self.net(recon)  # pass through SR network
            # target must match shape
            if u_lr.dim() == 4 and u_lr.size(0) == 1:
                target = u_lr.repeat(p_batch.size(0), 1, 1, 1)
            else:
                target = u_lr.repeat(p_batch.size(0), 1, 1, 1)
            # likelihood term (per-particle)
            loglik = -0.5 * ((pred - target) ** 2).mean() / (self.eps_lik ** 2)
            # Student-t prior log (up to constant)
            # logprior = -0.5 * (nu + 1) * mean(log1p(p^2 / (nu * lam)))
            logprior = -0.5 * (nu + 1.0) * torch.mean(torch.log1p((p_batch ** 2) / (nu * lam + 1e-12)))
            return loglik + self.prior_weight * logprior

        # run SVGD steps (the implementation mutates particles)
        for _ in range(self.svgd_steps):
            particles, velocity = svgd_step(particles, velocity, logpost_func,
                                            step_size=self.svgd_step_size,
                                            momentum=0.9, noise_scale=1e-6)
        # After SVGD, reconstruct HRs and return mean
        Yh_exp = [h.repeat(particles.size(0), *([1] * (h.dim() - 1))) for h in Yh]
        recon_particles = self.wavelet.idwt((particles, Yh_exp))  # (M, C, H_hr, W_hr)
        recon_mean = recon_particles.mean(0, keepdim=True)  # (1, C, H_hr, W_hr)
        # final pass through net (or could be identity depending on design). We keep minimal change:
        out = self.net(recon_mean)
        return out

    def state_dict(self, *args, **kwargs):
        st = super().state_dict(*args, **kwargs)
        # if trainable prior, include raw params so they are saved
        if self.trainable_prior:
            st['nu_param'] = self.nu_param.detach().cpu()
            st['lam_param'] = self.lam_param.detach().cpu()
        return st

    def load_state_dict(self, state_dict, strict=False):
        # allow loading nu/lam if present
        nu_present = 'nu_param' in state_dict
        lam_present = 'lam_param' in state_dict
        if nu_present and lam_present and self.trainable_prior:
            with torch.no_grad():
                self.nu_param.copy_(state_dict['nu_param'].to(self.nu_param.device))
                self.lam_param.copy_(state_dict['lam_param'].to(self.lam_param.device))
        # remove keys to avoid mismatch when calling super
        state_dict = {k: v for k, v in state_dict.items() if k not in ('nu_param', 'lam_param')}
        return super().load_state_dict(state_dict, strict=strict)
