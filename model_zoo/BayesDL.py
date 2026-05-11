from typing import *
"""
model_zoo/BayesDL.py: module for the Bayesian PWFEM / ProbSR project.
Auto-documented for open-source release.
"""

# -*- coding: utf-8 -*-
"""
BayesDL: Bayesian Deep Learning Super-Resolution Network (Fixed version)
修正版：
 - 修复 log_std / log_precision / log_df 输出分辨率错误问题
 - 改用动态 self.sr_scale 控制上采样倍数
 - 完全兼容原始接口与训练脚本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import conv, RCAB, Upsampler, ComplexConv2d


# ===============================
#        Residual Group (RG)
# ===============================
class ResidualGroup(nn.Module):
    def __init__(
        self, n_feat, kernel_size, n_resblocks,
        reduction=16, bias=True, bn=False, ln=False,
        act=nn.ReLU(True), res_scale=1, CA_type='CA',
        fft_branch='FFT_Layer', group_skip=True, block_skip=True
    ):
        super(ResidualGroup, self).__init__()
        self.skip = group_skip

        modules_body = [
            RCAB(
                n_feat, kernel_size, reduction=reduction, bias=bias,
                bn=bn, ln=ln, act=act, res_scale=res_scale,
                CA_type=CA_type, skip=block_skip, fft_branch=fft_branch
            )
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        if self.skip:
            res += x
        return res


# ===========================================
#  Bayesian Deep Learning Network (BayesDL)
# ===========================================
class BayesDL(nn.Module):
    def __init__(
        self,
        in_channels,
        sr_scale=2,
        out_channels=1,
        n_resgroups=10,
        n_resblocks=20,
        n_feat=64,
        kernel_size=3,
        reduction=16,
        act='relu',
        CA_type='CA',
        learn_std=False,
        drop=0.0,
        drop2d=0.0,
        fft_branch='',
        ln=False,
        fft_type='rfft',
        likelihood='gauss',
    ):
        super(BayesDL, self).__init__()

        self.in_channels = in_channels
        self.sr_scale = sr_scale
        self.learn_std = learn_std
        self.fft_type = fft_type
        self.likelihood = likelihood

        # activation
        if act.lower() == 'gelu':
            act = nn.GELU()
        else:
            act = nn.ReLU(True)

        # ============== Head ==============
        modules_head = [conv(in_channels, n_feat, kernel_size)]

        # ============== Body ==============
        modules_body = []
        for _ in range(n_resgroups):
            modules_body.append(
                ResidualGroup(
                    n_feat, kernel_size, n_resblocks=n_resblocks,
                    reduction=reduction, ln=ln, act=act,
                    CA_type=CA_type, fft_branch=fft_branch
                )
            )
            if drop > 0:
                modules_body.append(nn.Dropout(drop))
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        # ============== Tail ==============
        modules_tail = [
            Upsampler(sr_scale, n_feat, act=False),
            nn.Dropout2d(drop2d) if drop2d > 0 else nn.Identity(),
            conv(n_feat, out_channels, kernel_size)
        ]

        # ============== Uncertainty head ==============
        if self.learn_std:
            if self.likelihood in ['gauss', 'laplace']:
                self.uncer_tail = nn.Sequential(
                    conv(n_feat, n_feat, kernel_size), nn.ELU(),
                    conv(n_feat, n_feat, kernel_size), nn.ELU(),
                    conv(n_feat, n_feat, kernel_size), nn.ELU(),
                    conv(n_feat, n_feat, kernel_size), nn.ELU(),
                    conv(n_feat, 1, kernel_size)
                )
            elif self.likelihood == 'student':
                self.uncer_tail_precision = nn.Sequential(
                    conv(n_feat, n_feat, kernel_size), nn.ELU(),
                    conv(n_feat, n_feat, kernel_size), nn.ELU(),
                    conv(n_feat, n_feat, kernel_size), nn.ELU(),
                    conv(n_feat, n_feat, kernel_size), nn.ELU(),
                    conv(n_feat, 1, kernel_size)
                )
                self.uncer_tail_df = nn.Sequential(
                    conv(n_feat, n_feat, kernel_size), nn.ELU(),
                    conv(n_feat, n_feat, kernel_size), nn.ELU(),
                    conv(n_feat, n_feat, kernel_size), nn.ELU(),
                    conv(n_feat, n_feat, kernel_size), nn.ELU(),
                    conv(n_feat, 1, kernel_size)
                )

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    # ===========================================
    # Fixed Forward (2025)
    # ===========================================
    def forward(self, x):
        # ---- Encoder ----
        y = self.head(x)
        res = self.body(y)
        res += y

        # ---- Super-resolution reconstruction ----
        y_up = self.tail[0](res)
        y = self.tail[1:](y_up)  # final SR output

        # ---- Uncertainty branch ----
        if self.learn_std:
            scale = int(getattr(self, "sr_scale", 2))
            if scale < 1:
                scale = 1
            res_upsampled = F.interpolate(res, scale_factor=scale, mode="nearest")

            if self.likelihood in ['gauss', 'laplace']:
                log_std = self.uncer_tail(res_upsampled)
                return [y, log_std]

            elif self.likelihood == 'student':
                log_precision = self.uncer_tail_precision(res_upsampled)
                log_df = self.uncer_tail_df(res_upsampled)
                return [y, log_precision, log_df]

            else:
                raise ValueError(f"Unknown likelihood type: {self.likelihood}")

        # ---- Deterministic path ----
        else:
            return y

    # ===========================================
    # State dict loader
    # ===========================================
    def my_load_state_dict(self, state_dict, strict=True):
        own_state_dict = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state_dict:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state_dict[name].copy_(param)
                except Exception as e:
                    raise RuntimeError(
                        f"While copying parameter {name}, "
                        f"expected {own_state_dict[name].size()} but got {param.size()}"
                    ) from e
            elif strict:
                raise KeyError(f"Unexpected key {name} in state_dict")
