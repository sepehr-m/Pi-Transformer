import torch
import numpy as np
import logging
import os
import random


# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def compute_hurst_dfa(data, use_cpu=False):
    if use_cpu:
        data = data.cpu()
    else:
        data = data.to(
            torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        )

    if torch.isnan(data).any() or torch.isinf(data).any():
        logger.warning("NaN or Inf detected in input data for compute_hurst_dfa")
        # Replace NaN/Inf with zeros
        data = torch.where(
            torch.isnan(data) | torch.isinf(data), torch.zeros_like(data), data
        )

    N, L, C = data.shape
    hurst = torch.zeros(C, device=data.device)

    # DFA parameters
    min_scale = 4
    max_scale = L // 4
    if max_scale < min_scale:
        logger.warning(
            f"Sequence length {L} too short for DFA, returning default Hurst=0.5"
        )
        return torch.full((C,), 0.5, device=data.device)

    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num=10, dtype=int)
    scales = np.unique(scales)

    for c in range(C):
        channel_data = data[:, :, c]  # [N, L]
        fluctuations = []
        valid = True

        for scale in scales:
            # Segment the time series
            n_segments = L // scale
            if n_segments == 0:
                valid = False
                break

            segments = channel_data[:, : n_segments * scale].reshape(
                N, n_segments, scale
            )  # [N, n_segments, scale]

            t = torch.arange(1, scale + 1, device=data.device).float()  # [scale]
            cumsum = torch.cumsum(segments, dim=-1)  # [N, n_segments, scale]
            coeffs = torch_polyfit(t, cumsum, deg=1)  # Linear fit: [N, n_segments, 2]
            trend = coeffs[..., 0].unsqueeze(-1) * t + coeffs[..., 1].unsqueeze(
                -1
            )  # [N, n_segments, scale]
            detrended = cumsum - trend  # [N, n_segments, scale]

            # Compute fluctuation (RMS)
            fluctuation = torch.sqrt(
                torch.mean(detrended**2, dim=-1)
            )  # [N, n_segments]
            fluctuation = torch.mean(fluctuation)  # Scalar

            if torch.isnan(fluctuation) or torch.isinf(fluctuation):
                logger.warning(f"NaN/Inf fluctuation at channel {c}, scale {scale}")
                valid = False
                break
            fluctuations.append(fluctuation.item())

        if valid and len(fluctuations) > 1:

            log_scales = np.log(scales[: len(fluctuations)])
            log_fluctuations = np.log(np.array(fluctuations))
            if np.any(np.isnan(log_fluctuations)) or np.any(np.isinf(log_fluctuations)):
                logger.warning(f"NaN/Inf in log_fluctuations for channel {c}")
                hurst[c] = 0.5
            else:
                coeffs = np.polyfit(log_scales, log_fluctuations, 1)
                hurst[c] = coeffs[0]  # Slope is the Hurst exponent
                if hurst[c] <= 0 or hurst[c] >= 2 or np.isnan(hurst[c]):
                    logger.warning(
                        f"Invalid Hurst {hurst[c]} for channel {c}, setting to 0.5"
                    )
                    hurst[c] = 0.5
        else:
            logger.warning(f"Invalid DFA for channel {c}, setting Hurst to 0.5")
            hurst[c] = 0.5

    if torch.isnan(hurst).any():
        logger.warning("NaN detected in final Hurst values, replacing with 0.5")
        hurst = torch.where(torch.isnan(hurst), torch.full_like(hurst, 0.5), hurst)

    return hurst


def torch_polyfit(x, y, deg):
    device = y.device
    scale = x.shape[0]
    N, n_segments, _ = y.shape  # y: [N, n_segments, scale]

    powers = torch.arange(deg + 1, device=device).flip(0)  # [deg, deg-1, ..., 0]
    X = x.unsqueeze(-1).pow(powers)  # [scale, deg+1]

    y = y.unsqueeze(-1)  # [N, n_segments, scale, 1]

    X = X.view(1, 1, scale, deg + 1).expand(
        N, n_segments, -1, -1
    )  # [N, n_segments, scale, deg+1]


    coeffs = torch.linalg.lstsq(X, y).solution  # [N, n_segments, deg+1, 1]

    coeffs = coeffs.squeeze(-1)  # [N, n_segments, deg+1]

    if torch.isnan(coeffs).any():
        coeffs = torch.where(torch.isnan(coeffs), torch.zeros_like(coeffs), coeffs)

    return coeffs


def kl_loss(p, q):
    p = torch.clamp(p, min=1e-6, max=1.0)
    q = torch.clamp(q, min=1e-6, max=1.0)
    res = p * (torch.log(p) - torch.log(q))
    kl = torch.mean(torch.sum(res, dim=-1), dim=1)

    return kl


def adjust_learning_rate(optimiser, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimiser.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


# From Anomaly-Transformer Early stopping
# https://github.com/thuml/Anomaly-Transformer
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name="", delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_loss2_min = np.inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif (
            score < self.best_score + self.delta
            or score2 < self.best_score2 + self.delta
        ):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(
            model.state_dict(),
            os.path.join(path, str(self.dataset) + "_checkpoint.pth"),
        )
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2
