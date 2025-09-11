import torch
import torch.nn as nn
from math import sqrt
import numpy as np


class CausalMask:
    def __init__(self, B, L, device):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class PhaseSyncAttention(nn.Module):
    def __init__(
        self,
        win_size,
        mask_flag=True,
        scale=None,
        attention_dropout=0.0,
        output_attention=False,
        gamma=2.0,
        sigma=1.0,
        lambda_smooth=0.01,
        device="cuda",
    ):
        super(PhaseSyncAttention, self).__init__()
        self.device = torch.device(device)
        self.win_size = win_size
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention
        self.gamma = gamma
        self.sigma = sigma
        self.lambda_smooth = lambda_smooth

        indices = torch.arange(win_size, device=self.device)
        self.distances = torch.abs(indices.unsqueeze(1) - indices.unsqueeze(0))

    def compute_fractal_abscissa(self, hurst):
        hurst = torch.clamp(hurst, min=0.01, max=1.0)
        exp_hurst = torch.exp(self.gamma * hurst)
        psi = torch.cumsum(exp_hurst, dim=1)
        return psi

    def compute_gaussian_prior(self, psi):
        B, L, H = psi.shape

        psi_i = psi.unsqueeze(2)  # [B, L, 1, H]
        psi_j = psi.unsqueeze(1)  # [B, 1, L, H]
        diff = psi_i - psi_j  # [B, L, L, H]

        prior = torch.exp(-(diff**2) / (2 * self.sigma**2))
        prior = prior.permute(0, 3, 1, 2)  # [B, H, L, L]
        prior_sum = prior.sum(dim=-1, keepdim=True)
        prior = prior / (prior_sum + 1e-6)

        return prior

    def _hilbert_transform(self, signal):
        B, C, L = signal.shape
        h = torch.zeros(L, device=signal.device, dtype=signal.dtype)
        if L % 2 == 0:
            h[0] = h[L // 2] = 1.0
            h[1 : L // 2] = 2.0
        else:
            h[0] = 1.0
            h[1 : (L + 1) // 2] = 2.0

        return torch.fft.ifft(torch.fft.fft(signal, dim=-1) * h.view(1, 1, L), dim=-1)

    def compute_phase_gate(self, x, tau):
        B, L, D = x.shape
        analytic = self._hilbert_transform(x.permute(0, 2, 1))

        real_part = torch.real(analytic)
        imag_part = torch.imag(analytic)

        phase = torch.atan2(imag_part, real_part)
        theta = torch.atan2(torch.sin(phase).mean(1), torch.cos(phase).mean(1))

        dphi = torch.sin(0.5 * (theta.unsqueeze(2) - theta.unsqueeze(1)))
        tau = tau.unsqueeze(2).expand(-1, -1, L, -1)  # [B, L, L, H]
        S = torch.exp(-(dphi.unsqueeze(-1) ** 2) / (2 * tau**2))

        return S.permute(0, 3, 1, 2)  # [B, H, L, L]

    def compute_tau_smoothness_loss(self, tau):
        diff = tau[:, 1:, :] - tau[:, :-1, :]
        loss = self.lambda_smooth * torch.mean(diff**2)

        return loss

    def compute_hurst_smoothness_loss(self, hurst):
        diff = hurst[:, 1:, :] - hurst[:, :-1, :]
        loss = self.lambda_smooth * torch.mean(diff**2)

        return loss

    def compute_beta_prior_loss(self, hurst):
        alpha, beta = 2.0, 2.0
        hurst_clamped = torch.clamp(hurst, min=1e-6, max=1.0 - 1e-6)
        log_beta = (alpha - 1) * torch.log(hurst_clamped) + (beta - 1) * torch.log(
            1 - hurst_clamped
        )
        loss = -torch.mean(log_beta)

        return loss

    def forward(self, q, k, v, sigma, hurst, tau, attn_mask):
        B, L, H, E = q.shape
        _, S, _, D = v.shape

        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe, bshe->bhls", q, k)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = CausalMask(B, L, device=q.device)
            scores.masked_fill(attn_mask.mask, -np.inf)
        attn = scale * scores


        hurst = torch.sigmoid(hurst) * 0.99 + 0.01
        sigma = torch.sigmoid(sigma * 5) + 1e-6
        tau = torch.sigmoid(tau) * 0.9 + 0.1

        psi = self.compute_fractal_abscissa(hurst)
        prior = self.compute_gaussian_prior(psi)

        x_flat = q.permute(0, 2, 1, 3).reshape(B, L, -1).detach()
        gate = self.compute_phase_gate(x_flat, tau)
        prior = prior * gate
        prior = prior / (prior.sum(dim=-1, keepdim=True) + 1e-6)


        smoothness_loss = self.compute_hurst_smoothness_loss(hurst)
        tau_smoothness_loss = self.compute_tau_smoothness_loss(tau)
        beta_prior_loss = self.compute_beta_prior_loss(hurst)

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, v)

        if self.output_attention:
            return (
                V.contiguous(),
                series,
                prior,
                sigma,
                hurst,
                smoothness_loss,
                beta_prior_loss,
                tau_smoothness_loss,
            )
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_k=None, d_v=None):
        super(AttentionLayer, self).__init__()
        self.n_heads = n_heads
        d_k = d_k or (d_model // n_heads)
        d_v = d_v or (d_model // n_heads)

        self.LN = nn.LayerNorm(d_model)
        self.attention = attention

        self.q_projection = nn.Linear(d_model, d_k * n_heads)
        self.k_projection = nn.Linear(d_model, d_k * n_heads)
        self.v_projection = nn.Linear(d_model, d_model)

        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.hurst_projection = nn.Linear(d_model, n_heads)
        self.tau_projection = nn.Linear(d_model, n_heads)

        self.out_projection = nn.Linear(d_v * n_heads, d_model)

        # Initialise weights
        for m in [
            self.q_projection,
            self.k_projection,
            self.v_projection,
            self.sigma_projection,
            self.hurst_projection,
            self.tau_projection,
        ]:
            nn.init.xavier_uniform_(self.out_projection.weight)
            if self.out_projection.bias is not None:
                nn.init.zeros_(self.out_projection.bias)

    def forward(self, q, k, v, attn_mask):
        B, L, _ = q.shape
        _, S, _ = k.shape
        H = self.n_heads
        x = q

        q = self.q_projection(q).view(B, L, H, -1)
        k = self.k_projection(k).view(B, S, H, -1)
        v = self.v_projection(v).view(B, S, H, -1)

        sigma = self.sigma_projection(x).view(B, L, H)
        hurst = self.hurst_projection(x).view(B, L, H)
        tau = self.tau_projection(x).view(B, L, H)

        (
            out,
            series,
            prior,
            sigma,
            hurst,
            smoothness_loss,
            beta_prior_loss,
            tau_smoothness_loss,
        ) = self.attention(q, k, v, sigma, hurst, tau, attn_mask)

        out = out.reshape(B, L, -1)

        return (
            self.out_projection(out),
            series,
            prior,
            sigma,
            hurst,
            tau,
            smoothness_loss,
            beta_prior_loss,
            tau_smoothness_loss,
        )
