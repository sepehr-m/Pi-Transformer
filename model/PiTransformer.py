import torch
import torch.nn as nn
from model.attn import AttentionLayer, PhaseSyncAttention


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.LN = norm_layer

    def forward(self, x, attn_mask=None):
        series_list = []
        prior_list = []
        hurst_list = []
        sigma_list = []
        tau_list = []

        for i, attn_layer in enumerate(self.attn_layers):
            (
                x,
                series,
                prior,
                sigma,
                hurst,
                tau,
                smoothness_loss,
                beta_prior_loss,
                tau_smoothness_loss,
            ) = attn_layer(x, x, x, attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            hurst_list.append(hurst)
            sigma_list.append(sigma)
            tau_list.append(tau)

            if self.conv_layers is not None and i < len(self.conv_layers):
                x = x.permute(0, 2, 1)  # [B, D, L]
                x = self.conv_layers[i](x)  # [B, D, L]
                x = x.permute(0, 2, 1)  # [B, L, D]
        if self.LN is not None:
            x = self.LN(x)

        hurst_agg = torch.mean(torch.stack(hurst_list), dim=0)  # [B, L, H]
        tau_agg = torch.mean(torch.stack(tau_list), dim=0)  # [B, L, H]

        return (
            x,
            series_list,
            prior_list,
            sigma_list,
            hurst_agg,
            tau_agg,
            smoothness_loss,
            beta_prior_loss,
            tau_smoothness_loss,
        )


class PiTransformer(nn.Module):
    def __init__(
        self,
        win_size,
        enc_in,
        c_out,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_ff=512,
        dropout=0.0,
        activation="gelu",
        output_attention=True,
        gamma=2.0,
        sigma=1.0,
        lambda_smooth=0.01,
    ):
        super(PiTransformer, self).__init__()
        self.win_size = win_size
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, win_size, d_model))

        # Encoder
        attn_layers = [
            AttentionLayer(
                PhaseSyncAttention(
                    win_size,
                    mask_flag=True,
                    attention_dropout=dropout,
                    output_attention=output_attention,
                    gamma=gamma,
                    sigma=sigma,
                    lambda_smooth=lambda_smooth,
                ),
                d_model,
                n_heads,
            )
            for _ in range(e_layers - 1)
        ]
        conv_layers = [
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.ReLU() if activation == "relu" else nn.GELU(),
            )
            for _ in range(e_layers - 1)
        ]
        self.encoder = Encoder(
            attn_layers, conv_layers, norm_layer=nn.LayerNorm(d_model)
        )

        # Decoder
        self.decoder = nn.Linear(d_model, c_out)

    def forward(self, x):
        enc_in = self.enc_embedding(x) + self.enc_pos_embedding
        (
            enc_out,
            series,
            prior,
            sigma,
            hurst,
            tau,
            smoothness_loss,
            beta_prior_loss,
            tau_smoothness_loss,
        ) = self.encoder(enc_in, attn_mask=None)
        dec_out = self.decoder(enc_out)

        if self.output_attention:
            return (
                dec_out,
                series,
                prior,
                sigma,
                hurst,
                tau,
                smoothness_loss,
                beta_prior_loss,
                tau_smoothness_loss,
            )
        else:
            return dec_out
