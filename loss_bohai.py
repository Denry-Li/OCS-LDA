# loss_bohai.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_ORDER = [
    "inc_t2m",
    "inc_u10",
    "inc_v10",
    "inc_d2m",
    "inc_sp",
]


def get_increment_mean_std(norm_stats: dict, target_vars: list[str], device):
    means = []
    stds = []

    for var in target_vars:
        if var not in norm_stats:
            raise KeyError(f"{var} not found in norm_stats.")

        means.append(float(norm_stats[var]["mean"]))
        stds.append(float(norm_stats[var]["std"]))

    mean = torch.tensor(means, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std = torch.tensor(stds, dtype=torch.float32, device=device).view(1, -1, 1, 1)

    return mean, std


def denorm_increment(pred_inc_norm, inc_mean, inc_std):
    return pred_inc_norm * (inc_std + 1e-6) + inc_mean


def weighted_mse(pred, target, weight=None):
    err2 = (pred - target) ** 2

    if weight is None:
        return err2.mean()

    weight = weight.float()
    while weight.ndim < err2.ndim:
        weight = weight.unsqueeze(1)

    err2 = err2 * weight

    denom = weight.sum() * pred.shape[1]
    denom = denom.clamp_min(1.0)

    return err2.sum() / denom


def pred_loss(pred_inc_norm, target_inc_norm, variable_weights=None):
    """
    h芉S瀀蠎zz魰剉裿cw_c1Y0

    pred_inc_norm:
        [B, C, H, W]
    target_inc_norm:
        [B, C, H, W]
    variable_weights:
        optional tensor [C]
    """
    err2 = (pred_inc_norm - target_inc_norm) ** 2

    if variable_weights is not None:
        w = variable_weights.view(1, -1, 1, 1).to(err2.device)
        err2 = err2 * w

    return err2.mean()


def _make_gaussian_kernel(kernel_size, sigma, channels, device, dtype):
    if kernel_size % 2 == 0:
        raise ValueError("Gaussian kernel_size must be odd.")

    coords = torch.arange(kernel_size, device=device, dtype=dtype)
    coords = coords - (kernel_size - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    return kernel.repeat(channels, 1, 1, 1)


def gaussian_blur2d(x, kernel_size, sigma):
    channels = x.shape[1]
    kernel = _make_gaussian_kernel(
        kernel_size=int(kernel_size),
        sigma=float(sigma),
        channels=channels,
        device=x.device,
        dtype=x.dtype,
    )
    pad = int(kernel_size) // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x_pad, kernel, groups=channels)


def _channel_weighted_mean(loss_map, channel_weights=None):
    if channel_weights is None:
        return loss_map.mean()

    weights = channel_weights.to(loss_map.device, dtype=loss_map.dtype).view(1, -1, 1, 1)
    weighted = loss_map * weights
    denom = weights.sum().clamp_min(1e-12)
    denom = denom * loss_map.shape[0] * loss_map.shape[2] * loss_map.shape[3]
    return weighted.sum() / denom


class NormalizedPMLFLoss(nn.Module):
    """
    Normalized PMLF-style loss for increment-learning assimilation.

    The main Huber term is computed in normalized increment space. The
    high/low-frequency structure terms are computed in physical analysis space:
        analysis = raw_gfs + denormalized_increment
    """

    def __init__(self, cfg, inc_mean, inc_std, target_vars):
        super().__init__()
        self.target_vars = list(target_vars)

        self.lambda_ms = float(cfg.get("lambda_ms", 0.1))
        self.huber_delta = float(cfg.get("pmlf_huber_delta", 0.05))

        self.high_kernel_size = int(cfg.get("pmlf_high_kernel_size", 5))
        self.high_sigma = float(cfg.get("pmlf_high_sigma", 1.0))
        self.low_kernel_size = int(cfg.get("pmlf_low_kernel_size", 9))
        self.low_sigma = float(cfg.get("pmlf_low_sigma", 2.0))

        self.dynamic_alpha = bool(cfg.get("pmlf_dynamic_alpha", True))
        self.beta = float(cfg.get("pmlf_beta", 1.0))
        self.ema_momentum = float(cfg.get("pmlf_ema_momentum", 0.99))
        self.alpha_min = float(cfg.get("pmlf_alpha_min", 0.2))
        self.eps = float(cfg.get("pmlf_eps", 1e-8))

        self.fixed_alpha_high = float(cfg.get("pmlf_alpha_high", 0.5))
        self.fixed_alpha_low = float(cfg.get("pmlf_alpha_low", 0.5))

        self.register_buffer("inc_mean", inc_mean.detach().clone())
        self.register_buffer("inc_std", inc_std.detach().clone())
        self.register_buffer("ema_high", torch.tensor(0.0))
        self.register_buffer("ema_low", torch.tensor(0.0))
        self.register_buffer("ema_initialized", torch.tensor(False, dtype=torch.bool))

        high_weights = cfg.get("pmlf_high_variable_weights", None)
        low_weights = cfg.get("pmlf_low_variable_weights", None)
        high_weights_tensor = self._build_weights(high_weights)
        low_weights_tensor = self._build_weights(low_weights)
        self.register_buffer("high_variable_weights", high_weights_tensor)
        self.register_buffer("low_variable_weights", low_weights_tensor)

    def _build_weights(self, weights):
        if weights is None:
            return None
        if len(weights) != len(self.target_vars):
            raise ValueError("PMLF variable weights must match target_vars length.")
        return torch.tensor(weights, dtype=torch.float32)

    def _analysis_fields(self, pred_inc_norm, target_inc_norm, batch):
        if "raw_gfs" not in batch:
            raise KeyError("Batch missing raw_gfs, required by pmlf loss.")

        raw_gfs = batch["raw_gfs"].to(
            pred_inc_norm.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        pred_inc = denorm_increment(pred_inc_norm.float(), self.inc_mean, self.inc_std)
        target_inc = denorm_increment(target_inc_norm.float(), self.inc_mean, self.inc_std)
        return raw_gfs + pred_inc, raw_gfs + target_inc

    def _alphas(self, loss_high, loss_low):
        if not self.dynamic_alpha:
            alpha_high = torch.tensor(self.fixed_alpha_high, device=loss_high.device)
            alpha_low = torch.tensor(self.fixed_alpha_low, device=loss_low.device)
            denom = (alpha_high + alpha_low).clamp_min(self.eps)
            return alpha_high / denom, alpha_low / denom

        with torch.no_grad():
            high_detached = loss_high.detach()
            low_detached = loss_low.detach()

            if not bool(self.ema_initialized.item()):
                self.ema_high.copy_(high_detached)
                self.ema_low.copy_(low_detached)
                self.ema_initialized.fill_(True)
            elif torch.is_grad_enabled():
                self.ema_high.mul_(self.ema_momentum).add_(
                    high_detached * (1.0 - self.ema_momentum)
                )
                self.ema_low.mul_(self.ema_momentum).add_(
                    low_detached * (1.0 - self.ema_momentum)
                )

            r_high = high_detached / (self.ema_high + self.eps)
            r_low = low_detached / (self.ema_low + self.eps)
            logits = torch.stack([r_high, r_low]) * self.beta
            raw = torch.softmax(logits, dim=0)
            if self.alpha_min > 0:
                raw = self.alpha_min + (1.0 - 2.0 * self.alpha_min) * raw

        return raw[0].to(loss_high.device), raw[1].to(loss_low.device)

    def forward(self, pred_inc_norm, target_inc_norm, batch):
        loss_inc = F.huber_loss(
            pred_inc_norm.float(),
            target_inc_norm.float(),
            delta=self.huber_delta,
            reduction="mean",
        )

        pred_analysis, target_analysis = self._analysis_fields(
            pred_inc_norm=pred_inc_norm,
            target_inc_norm=target_inc_norm,
            batch=batch,
        )

        pred_high = pred_analysis - gaussian_blur2d(
            pred_analysis,
            self.high_kernel_size,
            self.high_sigma,
        )
        target_high = target_analysis - gaussian_blur2d(
            target_analysis,
            self.high_kernel_size,
            self.high_sigma,
        )
        loss_high = _channel_weighted_mean(
            (pred_high - target_high) ** 2,
            self.high_variable_weights,
        )

        pred_low = gaussian_blur2d(pred_analysis, self.low_kernel_size, self.low_sigma)
        target_low = gaussian_blur2d(target_analysis, self.low_kernel_size, self.low_sigma)
        loss_low = _channel_weighted_mean(
            torch.log1p(torch.abs(pred_low - target_low)),
            self.low_variable_weights,
        )

        alpha_high, alpha_low = self._alphas(loss_high, loss_low)
        loss_ms = alpha_high * loss_high + alpha_low * loss_low
        loss_total = loss_inc + self.lambda_ms * loss_ms

        logs = {
            "loss_total": loss_total.detach(),
            "loss_pred": loss_inc.detach(),
            "loss_inc": loss_inc.detach(),
            "loss_high": loss_high.detach(),
            "loss_low": loss_low.detach(),
            "loss_ms": loss_ms.detach(),
            "alpha_high": alpha_high.detach(),
            "alpha_low": alpha_low.detach(),
            "ema_high": self.ema_high.detach(),
            "ema_low": self.ema_low.detach(),
        }
        return loss_total, logs


class AEReconstructionLoss(nn.Module):
    """
    Reconstruction loss for standardized ERA5 autoencoder training.

    ae_loss_mode=mse:
        L = MSE(pred, target) in standardized ERA5 space.

    ae_loss_mode=pmlf:
        L = L_rec + lambda_ms * (alpha_high * L_high + alpha_low * L_low)
        L_rec is computed in standardized ERA5 space, and high/low-frequency
        structure terms are computed in denormalized physical ERA5 space.
    """

    def __init__(self, cfg, field_mean, field_std, target_vars):
        super().__init__()
        self.target_vars = list(target_vars)
        self.use_branch_aux_loss = bool(cfg.get("use_branch_aux_loss", False)) or float(
            cfg.get("lambda_branch", 0.0)
        ) > 0
        self.lambda_branch = float(cfg.get("lambda_branch", 0.0))
        self.branch_loss_weights = {
            str(k): float(v)
            for k, v in dict(cfg.get("branch_loss_weights", {})).items()
        }
        self.aux_variable_channels = {
            str(k): [int(i) for i in v]
            for k, v in dict(
                cfg.get(
                    "aux_variable_channels",
                    {
                        "temp": [0],
                        "wind": [1, 2],
                        "moist": [3],
                        "press": [4],
                    },
                )
            ).items()
        }
        orth_cfg = dict(cfg.get("orth_loss", {}) or {})
        self.lambda_orth = float(orth_cfg.get("weight", cfg.get("lambda_orth", 0.0)))
        self.use_orth_loss = bool(orth_cfg.get("enabled", False)) or self.lambda_orth > 0.0
        self.orth_branches = [
            str(name)
            for name in orth_cfg.get(
                "branches",
                cfg.get("orth_branches", ["wind", "temp", "moist", "press"]),
            )
        ]
        self.orth_center = bool(orth_cfg.get("center", True))
        self.orth_normalize = bool(orth_cfg.get("normalize", True))
        self.orth_descriptor = str(orth_cfg.get("descriptor", "channel_mean")).lower()

        self.lambda_tp_log = float(cfg.get("lambda_tp_log", 0.0))
        self.tp_log_var = str(cfg.get("tp_log_var", "tp"))
        self.tp_log_scale = float(cfg.get("tp_log_scale", 0.001))
        self.tp_log_softplus_scale = float(
            cfg.get("tp_log_softplus_scale", self.tp_log_scale)
        )
        if self.tp_log_scale <= 0.0 or self.tp_log_softplus_scale <= 0.0:
            raise ValueError("tp_log_scale and tp_log_softplus_scale must be positive.")
        if self.lambda_tp_log > 0.0 and self.tp_log_var not in self.target_vars:
            raise ValueError(f"tp_log_var={self.tp_log_var!r} is not in target_vars.")
        self.tp_log_channel = (
            self.target_vars.index(self.tp_log_var) if self.tp_log_var in self.target_vars else None
        )
        if "ae_loss_mode" in cfg:
            mode = str(cfg["ae_loss_mode"]).lower()
        else:
            legacy_rec = str(cfg.get("ae_rec_loss", "huber")).lower()
            legacy_lambda_ms = float(cfg.get("lambda_ms", 0.1))
            mode = "mse" if legacy_rec == "mse" and legacy_lambda_ms <= 0 else "pmlf"
        self.mode = mode
        if self.mode in {"pure_mse", "mse_only"}:
            self.mode = "mse"
        if self.mode in {"normalized_pmlf", "full_pmlf"}:
            self.mode = "pmlf"
        if self.mode not in {"mse", "pmlf"}:
            raise ValueError("ae_loss_mode must be 'mse' or 'pmlf'.")

        default_rec = "mse" if self.mode == "mse" else "huber"
        self.rec_loss = str(cfg.get("ae_rec_loss", default_rec)).lower()
        if self.mode == "mse":
            self.rec_loss = "mse"

        self.huber_delta = float(cfg.get("ae_huber_delta", cfg.get("pmlf_huber_delta", 0.05)))
        self.lambda_ms = float(cfg.get("lambda_ms", 0.1 if self.mode == "pmlf" else 0.0))
        self.high_kernel_size = int(cfg.get("ae_high_kernel_size", cfg.get("pmlf_high_kernel_size", 5)))
        self.high_sigma = float(cfg.get("ae_high_sigma", cfg.get("pmlf_high_sigma", 1.0)))
        self.low_kernel_size = int(cfg.get("ae_low_kernel_size", cfg.get("pmlf_low_kernel_size", 9)))
        self.low_sigma = float(cfg.get("ae_low_sigma", cfg.get("pmlf_low_sigma", 2.0)))

        self.dynamic_alpha = bool(cfg.get("ae_dynamic_alpha", cfg.get("pmlf_dynamic_alpha", True)))
        self.beta = float(cfg.get("ae_beta", cfg.get("pmlf_beta", 1.0)))
        self.ema_momentum = float(cfg.get("ae_ema_momentum", cfg.get("pmlf_ema_momentum", 0.99)))
        self.alpha_min = float(cfg.get("ae_alpha_min", cfg.get("pmlf_alpha_min", 0.2)))
        self.eps = float(cfg.get("ae_eps", cfg.get("pmlf_eps", 1e-8)))

        self.fixed_alpha_high = float(cfg.get("ae_alpha_high", cfg.get("pmlf_alpha_high", 0.5)))
        self.fixed_alpha_low = float(cfg.get("ae_alpha_low", cfg.get("pmlf_alpha_low", 0.5)))

        high_weights = cfg.get("ae_high_variable_weights", cfg.get("pmlf_high_variable_weights", None))
        low_weights = cfg.get("ae_low_variable_weights", cfg.get("pmlf_low_variable_weights", None))
        high_weights_tensor = self._build_weights(high_weights)
        low_weights_tensor = self._build_weights(low_weights)

        self.register_buffer("field_mean", field_mean.detach().clone())
        self.register_buffer("field_std", field_std.detach().clone())
        self.register_buffer("high_variable_weights", high_weights_tensor)
        self.register_buffer("low_variable_weights", low_weights_tensor)
        self.register_buffer("ema_high", torch.tensor(0.0))
        self.register_buffer("ema_low", torch.tensor(0.0))
        self.register_buffer("ema_initialized", torch.tensor(False, dtype=torch.bool))

    def _build_weights(self, weights):
        if weights is None:
            return None
        if len(weights) != len(self.target_vars):
            raise ValueError("AE PMLF variable weights must match target_vars length.")
        return torch.tensor(weights, dtype=torch.float32)

    def _to_physical(self, x):
        return x.float() * (self.field_std + 1e-6) + self.field_mean

    def _alphas(self, loss_high, loss_low):
        if not self.dynamic_alpha:
            alpha_high = torch.tensor(
                self.fixed_alpha_high,
                device=loss_high.device,
                dtype=loss_high.dtype,
            )
            alpha_low = torch.tensor(
                self.fixed_alpha_low,
                device=loss_low.device,
                dtype=loss_low.dtype,
            )
            denom = (alpha_high + alpha_low).clamp_min(self.eps)
            return alpha_high / denom, alpha_low / denom

        with torch.no_grad():
            high_detached = loss_high.detach()
            low_detached = loss_low.detach()

            if not bool(self.ema_initialized.item()):
                self.ema_high.copy_(high_detached)
                self.ema_low.copy_(low_detached)
                self.ema_initialized.fill_(True)
            elif torch.is_grad_enabled():
                self.ema_high.mul_(self.ema_momentum).add_(
                    high_detached * (1.0 - self.ema_momentum)
                )
                self.ema_low.mul_(self.ema_momentum).add_(
                    low_detached * (1.0 - self.ema_momentum)
                )

            r_high = high_detached / (self.ema_high + self.eps)
            r_low = low_detached / (self.ema_low + self.eps)
            logits = torch.stack([r_high, r_low]) * self.beta
            raw = torch.softmax(logits, dim=0)
            if self.alpha_min > 0:
                raw = self.alpha_min + (1.0 - 2.0 * self.alpha_min) * raw

        return raw[0].to(loss_high.device), raw[1].to(loss_low.device)

    def _tp_log_aux_loss(self, pred, target):
        if self.lambda_tp_log <= 0.0 or self.tp_log_channel is None:
            zero = torch.tensor(0.0, device=target.device, dtype=target.dtype)
            return zero

        idx = int(self.tp_log_channel)
        pred_phys = (
            pred[:, idx].float() * (self.field_std[:, idx].float() + 1e-6)
            + self.field_mean[:, idx].float()
        )
        target_phys = (
            target[:, idx].float() * (self.field_std[:, idx].float() + 1e-6)
            + self.field_mean[:, idx].float()
        )

        soft_scale = float(self.tp_log_softplus_scale)
        pred_nonneg = F.softplus(pred_phys / soft_scale) * soft_scale
        target_nonneg = target_phys.clamp_min(0.0)

        log_scale = float(self.tp_log_scale)
        pred_log = torch.log1p(pred_nonneg / log_scale)
        target_log = torch.log1p(target_nonneg / log_scale)
        return F.mse_loss(pred_log, target_log)

    def _branch_aux_loss(self, pred_dict, target):
        if not isinstance(pred_dict, dict) or "aux" not in pred_dict:
            zero = torch.tensor(0.0, device=target.device, dtype=target.dtype)
            return zero, {}

        loss_total = torch.tensor(0.0, device=target.device, dtype=target.dtype)
        logs = {}
        for name, pred_aux in pred_dict["aux"].items():
            if name not in self.aux_variable_channels:
                continue
            indices = self.aux_variable_channels[name]
            target_aux = target[:, indices].float()
            loss_i = F.mse_loss(pred_aux.float(), target_aux)
            weight = float(self.branch_loss_weights.get(name, 1.0))
            loss_total = loss_total + weight * loss_i
            logs[f"loss_aux_{name}"] = loss_i.detach()
        return loss_total, logs

    def _branch_orthogonality_loss(self, pred_dict, target):
        if not self.use_orth_loss:
            zero = torch.tensor(0.0, device=target.device, dtype=target.dtype)
            return zero, {}
        if not isinstance(pred_dict, dict) or "z_dict" not in pred_dict:
            zero = torch.tensor(0.0, device=target.device, dtype=target.dtype)
            return zero, {}

        z_dict = pred_dict["z_dict"]
        descriptors = {}
        for name in self.orth_branches:
            if name not in z_dict:
                continue
            z = z_dict[name].float()
            if self.orth_descriptor == "channel_mean":
                desc = z.mean(dim=1).flatten(start_dim=1)
            elif self.orth_descriptor == "channel_rms":
                desc = torch.sqrt(torch.mean(z * z, dim=1).clamp_min(self.eps)).flatten(start_dim=1)
            else:
                raise ValueError("orth_loss.descriptor must be 'channel_mean' or 'channel_rms'.")
            if self.orth_center:
                desc = desc - desc.mean(dim=1, keepdim=True)
            if self.orth_normalize:
                desc = F.normalize(desc, dim=1, eps=self.eps)
            descriptors[name] = desc

        names = [name for name in self.orth_branches if name in descriptors]
        if len(names) < 2:
            zero = torch.tensor(0.0, device=target.device, dtype=target.dtype)
            return zero, {}

        losses = []
        logs = {}
        for i, name_i in enumerate(names):
            for name_j in names[i + 1:]:
                cos = torch.sum(descriptors[name_i] * descriptors[name_j], dim=1)
                pair_loss = torch.mean(cos * cos)
                losses.append(pair_loss)
                logs[f"orth_cos2_{name_i}_{name_j}"] = pair_loss.detach()
                logs[f"orth_abs_cos_{name_i}_{name_j}"] = torch.mean(torch.abs(cos)).detach()
        return torch.stack(losses).mean(), logs

    def forward(self, pred, target):
        pred_dict = pred if isinstance(pred, dict) else None
        if isinstance(pred, dict):
            if "full" not in pred:
                raise KeyError("AE reconstruction dict output must include key 'full'.")
            pred = pred["full"]
        pred = pred.float()
        target = target.float()

        if self.rec_loss == "mse":
            loss_rec = F.mse_loss(pred, target)
        elif self.rec_loss == "huber":
            loss_rec = F.huber_loss(
                pred,
                target,
                delta=self.huber_delta,
                reduction="mean",
            )
        else:
            raise ValueError("ae_rec_loss must be 'mse' or 'huber'.")

        if self.mode == "mse":
            zero = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            loss_branch, branch_logs = self._branch_aux_loss(pred_dict, target)
            loss_orth, orth_logs = self._branch_orthogonality_loss(pred_dict, target)
            loss_tp_log = self._tp_log_aux_loss(pred, target)
            loss_total = (
                loss_rec
                + self.lambda_branch * loss_branch
                + self.lambda_orth * loss_orth
                + self.lambda_tp_log * loss_tp_log
            )
            logs = {
                "loss_total": loss_total.detach(),
                "loss_pred": loss_rec.detach(),
                "loss_rec": loss_rec.detach(),
                "loss_branch": loss_branch.detach(),
                "loss_orth": loss_orth.detach(),
                "loss_tp_log": loss_tp_log.detach(),
                "lambda_orth": torch.tensor(self.lambda_orth, device=pred.device).detach(),
                "lambda_tp_log": torch.tensor(self.lambda_tp_log, device=pred.device).detach(),
                "loss_high": zero.detach(),
                "loss_low": zero.detach(),
                "loss_ms": zero.detach(),
                "alpha_high": zero.detach(),
                "alpha_low": zero.detach(),
                "ema_high": self.ema_high.detach(),
                "ema_low": self.ema_low.detach(),
            }
            logs.update(branch_logs)
            logs.update(orth_logs)
            return loss_total, logs

        pred_phys = self._to_physical(pred)
        target_phys = self._to_physical(target)

        pred_high = pred_phys - gaussian_blur2d(
            pred_phys,
            self.high_kernel_size,
            self.high_sigma,
        )
        target_high = target_phys - gaussian_blur2d(
            target_phys,
            self.high_kernel_size,
            self.high_sigma,
        )
        loss_high = _channel_weighted_mean(
            (pred_high - target_high) ** 2,
            self.high_variable_weights,
        )

        pred_low = gaussian_blur2d(pred_phys, self.low_kernel_size, self.low_sigma)
        target_low = gaussian_blur2d(target_phys, self.low_kernel_size, self.low_sigma)
        loss_low = _channel_weighted_mean(
            torch.log1p(torch.abs(pred_low - target_low)),
            self.low_variable_weights,
        )

        alpha_high, alpha_low = self._alphas(loss_high, loss_low)
        loss_ms = alpha_high * loss_high + alpha_low * loss_low
        loss_branch, branch_logs = self._branch_aux_loss(pred_dict, target)
        loss_orth, orth_logs = self._branch_orthogonality_loss(pred_dict, target)
        loss_tp_log = self._tp_log_aux_loss(pred, target)
        loss_total = (
            loss_rec
            + self.lambda_ms * loss_ms
            + self.lambda_branch * loss_branch
            + self.lambda_orth * loss_orth
            + self.lambda_tp_log * loss_tp_log
        )

        logs = {
            "loss_total": loss_total.detach(),
            "loss_pred": loss_rec.detach(),
            "loss_rec": loss_rec.detach(),
            "loss_branch": loss_branch.detach(),
            "loss_orth": loss_orth.detach(),
            "loss_tp_log": loss_tp_log.detach(),
            "lambda_orth": torch.tensor(self.lambda_orth, device=pred.device).detach(),
            "lambda_tp_log": torch.tensor(self.lambda_tp_log, device=pred.device).detach(),
            "loss_high": loss_high.detach(),
            "loss_low": loss_low.detach(),
            "loss_ms": loss_ms.detach(),
            "alpha_high": alpha_high.detach(),
            "alpha_low": alpha_low.detach(),
            "ema_high": self.ema_high.detach(),
            "ema_low": self.ema_low.detach(),
        }
        logs.update(branch_logs)
        logs.update(orth_logs)
        return loss_total, logs


class StructuredAEReconstructionLoss(nn.Module):
    """
    Training loss adapter for models.StructedSwimAE.StructuredSwinAE.

    The model returns a dictionary with x_hat, x_base_low, and branches when
    return_parts=True. This adapter delegates the actual structured terms to
    models.StructedSwimAE.structured_ae_loss and normalizes the log names used
    by train_bohai.py.
    """

    def __init__(self, cfg, target_vars: list[str]):
        super().__init__()
        self.cfg = cfg
        self.target_vars = target_vars

        weights = cfg.get("structured_var_weights", cfg.get("variable_loss_weights", None))
        if weights is not None:
            if len(weights) != len(target_vars):
                raise ValueError(
                    "structured_var_weights/variable_loss_weights length must "
                    "match target_vars length."
                )
            self.register_buffer(
                "var_weights",
                torch.tensor(weights, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.var_weights = None

        self.loss_mode = str(
            cfg.get("structured_loss_mode", cfg.get("ae_loss_mode", "huber"))
        ).lower()
        self.huber_delta = float(
            cfg.get("structured_huber_delta", cfg.get("ae_huber_delta", 1.0))
        )
        self.lambda_low = float(cfg.get("lambda_structured_low", cfg.get("lambda_low", 0.1)))
        self.lambda_orth = float(
            cfg.get("lambda_structured_orth", cfg.get("lambda_orth", 1e-3))
        )
        self.low_kernel_size = int(
            cfg.get("structured_low_kernel_size", cfg.get("low_kernel_size", 5))
        )
        self.lowpass_type = str(
            cfg.get("structured_lowpass_type", cfg.get("lowpass_type", "gaussian"))
        ).lower()
        self.low_sigma = float(cfg.get("structured_low_sigma", cfg.get("low_sigma", 1.0)))

    def forward(self, pred, target):
        from models.StructedSwimAE import StructuredSwinAE, structured_ae_loss

        if not isinstance(pred, dict):
            raise TypeError(
                "StructuredAEReconstructionLoss expects StructedSwimAE output "
                "dict. Ensure forward_model calls model(..., return_parts=True)."
            )

        loss_dict = structured_ae_loss(
            model=StructuredSwinAE,
            output=pred,
            target=target,
            var_weights=self.var_weights,
            loss_type=self.loss_mode,
            huber_delta=self.huber_delta,
            lambda_low=self.lambda_low,
            lambda_orth=self.lambda_orth,
            low_kernel_size=self.low_kernel_size,
            lowpass_type=self.lowpass_type,
            low_sigma=self.low_sigma,
        )

        loss_total = loss_dict["loss"]
        logs = {
            "loss_pred": loss_dict["L_recon"].detach(),
            "loss_structured_low": loss_dict["L_low"].detach(),
            "loss_structured_orth": loss_dict["L_orth"].detach(),
            "loss_total": loss_total.detach(),
        }
        return loss_total, logs


def observation_loss_ccmp(
    pred_inc_norm,
    raw_gfs,
    ccmp_u10,
    ccmp_v10,
    ccmp_mask,
    ccmp_conf,
    inc_mean,
    inc_std,
    u10_channel=1,
    v10_channel=2,
):
    """
    CCMP 聣KmN魜'`_c1Y0

    闟_g u10 / v100
    (Wirtzz魰梴
        analysis = GFS + pred_inc

    raw_gfs:
        [B, 5, H, W]
    ccmp_u10, ccmp_v10, ccmp_mask, ccmp_conf:
        [B, H, W]
    """
    pred_inc = denorm_increment(pred_inc_norm, inc_mean, inc_std)
    analysis = raw_gfs + pred_inc

    analysis_u10 = analysis[:, u10_channel]
    analysis_v10 = analysis[:, v10_channel]

    obs_weight = ccmp_mask.float() * ccmp_conf.float()
    obs_weight = obs_weight.clamp_min(0.0)

    denom = obs_weight.sum().clamp_min(1.0)

    loss_u = (((analysis_u10 - ccmp_u10) ** 2) * obs_weight).sum() / denom
    loss_v = (((analysis_v10 - ccmp_v10) ** 2) * obs_weight).sum() / denom

    return loss_u + loss_v


def build_coast_smooth_weight(
    signed_distance_to_coast_km,
    nearshore_km=50.0,
    nearshore_weight=0.3,
    offshore_weight=1.0,
):
    """
    刧 ?coast-aware s^裯Cg蛻0

    褟竆:SM朜Os^裯:_
怣Q筨s^w瀃wmF柉h0

    signed_distance_to_coast_km:
        [B, H, W]
    """
    abs_dist = torch.abs(signed_distance_to_coast_km)

    weight = torch.full_like(abs_dist, fill_value=offshore_weight)
    weight = torch.where(
        abs_dist <= nearshore_km,
        torch.full_like(weight, fill_value=nearshore_weight),
        weight,
    )

    return weight


def smoothness_loss(
    pred_inc_norm,
    inc_mean,
    inc_std,
    signed_distance_to_coast_km=None,
    smooth_channels=None,
    coast_aware=True,
    nearshore_km=50.0,
    nearshore_weight=0.3,
):
    """
    瀀蠎s^裯_c1Y0

    貫鵞irtzz魰 pred_inc 梴痟s^裯

€
N/f鵞h芉Szz魰梴0

    pred_inc_norm:
        [B, C, H, W]

    smooth_channels:
        梺塻^裯剉怱?index0
        None h?y@b	g豐蠎龕s^裯0
        ╟P?R薡O(u [1, 2, 4]
sS u10/v10/sp0
    """
    pred_inc = denorm_increment(pred_inc_norm, inc_mean, inc_std)

    if smooth_channels is not None:
        pred_inc = pred_inc[:, smooth_channels]

    dx = pred_inc[..., :, 1:] - pred_inc[..., :, :-1]
    dy = pred_inc[..., 1:, :] - pred_inc[..., :-1, :]

    if coast_aware and signed_distance_to_coast_km is not None:
        weight = build_coast_smooth_weight(
            signed_distance_to_coast_km=signed_distance_to_coast_km,
            nearshore_km=nearshore_km,
            nearshore_weight=nearshore_weight,
            offshore_weight=1.0,
        )

        wx = weight[..., :, 1:]
        wy = weight[..., 1:, :]

        while wx.ndim < dx.ndim:
            wx = wx.unsqueeze(1)
        while wy.ndim < dy.ndim:
            wy = wy.unsqueeze(1)

        loss_x = (dx**2 * wx).mean()
        loss_y = (dy**2 * wy).mean()
    else:
        loss_x = (dx**2).mean()
        loss_y = (dy**2).mean()

    return loss_x + loss_y


class BohaiCompositeLoss:
    """
    Composite loss:

        L = L_pred + lambda_obs * L_obs + lambda_smooth * L_smooth
    """

    def __init__(self, cfg, norm_stats: dict, device):
        self.cfg = cfg
        self.target_vars = list(cfg["target_vars"])
        self.norm_stats = norm_stats
        self.device = device

        self.lambda_obs = float(cfg.get("lambda_obs", 0.0))
        self.lambda_smooth = float(cfg.get("lambda_smooth", 0.0))

        self.use_obs_loss = bool(cfg.get("use_obs_loss", False))
        self.use_smooth_loss = bool(cfg.get("use_smooth_loss", False))
        self.loss_type = str(cfg.get("loss_type", "pred")).lower()

        self.inc_mean, self.inc_std = get_increment_mean_std(
            norm_stats=norm_stats,
            target_vars=self.target_vars,
            device=device,
        )

        variable_weights = cfg.get("variable_loss_weights", None)
        if variable_weights is not None:
            if len(variable_weights) != len(self.target_vars):
                raise ValueError(
                    "variable_loss_weights length must match target_vars length."
                )
            self.variable_weights = torch.tensor(
                variable_weights,
                dtype=torch.float32,
                device=device,
            )
        else:
            self.variable_weights = None

        smooth_channels = cfg.get("smooth_channels", None)
        if smooth_channels is None:
            self.smooth_channels = None
        else:
            self.smooth_channels = [int(x) for x in smooth_channels]

        self.pmlf_loss = None
        self.ae_loss = None
        self.structured_ae_loss = None
        if self.loss_type == "pmlf":
            self.pmlf_loss = NormalizedPMLFLoss(
                cfg=cfg,
                inc_mean=self.inc_mean,
                inc_std=self.inc_std,
                target_vars=self.target_vars,
            ).to(device)
        elif self.loss_type == "ae_recon":
            self.ae_loss = AEReconstructionLoss(
                cfg=cfg,
                field_mean=self.inc_mean,
                field_std=self.inc_std,
                target_vars=self.target_vars,
            ).to(device)
        elif self.loss_type == "structured_ae":
            self.structured_ae_loss = StructuredAEReconstructionLoss(
                cfg=cfg,
                target_vars=self.target_vars,
            ).to(device)

    def __call__(self, pred_inc_norm, target_inc_norm, batch):
        if self.structured_ae_loss is not None:
            return self.structured_ae_loss(pred_inc_norm, target_inc_norm)

        if self.ae_loss is not None:
            return self.ae_loss(pred_inc_norm, target_inc_norm)

        if self.pmlf_loss is not None:
            return self.pmlf_loss(pred_inc_norm, target_inc_norm, batch)

        loss_pred = pred_loss(
            pred_inc_norm=pred_inc_norm,
            target_inc_norm=target_inc_norm,
            variable_weights=self.variable_weights,
        )

        loss_obs = torch.tensor(0.0, device=pred_inc_norm.device)
        loss_smooth = torch.tensor(0.0, device=pred_inc_norm.device)

        if self.use_obs_loss and self.lambda_obs > 0:
            required = [
                "raw_gfs",
                "ccmp_u10",
                "ccmp_v10",
                "ccmp_mask",
                "ccmp_conf",
            ]
            for key in required:
                if key not in batch:
                    raise KeyError(f"Batch missing key for obs loss: {key}")

            raw_gfs = batch["raw_gfs"].to(
                pred_inc_norm.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            ccmp_u10 = batch["ccmp_u10"].to(
                pred_inc_norm.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            ccmp_v10 = batch["ccmp_v10"].to(
                pred_inc_norm.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            ccmp_mask = batch["ccmp_mask"].to(
                pred_inc_norm.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            ccmp_conf = batch["ccmp_conf"].to(
                pred_inc_norm.device,
                dtype=torch.float32,
                non_blocking=True,
            )

            loss_obs = observation_loss_ccmp(
                pred_inc_norm=pred_inc_norm,
                raw_gfs=raw_gfs,
                ccmp_u10=ccmp_u10,
                ccmp_v10=ccmp_v10,
                ccmp_mask=ccmp_mask,
                ccmp_conf=ccmp_conf,
                inc_mean=self.inc_mean,
                inc_std=self.inc_std,
                u10_channel=1,
                v10_channel=2,
            )

        if self.use_smooth_loss and self.lambda_smooth > 0:
            signed_distance_to_coast_km = None

            if "signed_distance_to_coast_km" in batch:
                signed_distance_to_coast_km = batch["signed_distance_to_coast_km"].to(
                    pred_inc_norm.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )

            loss_smooth = smoothness_loss(
                pred_inc_norm=pred_inc_norm,
                inc_mean=self.inc_mean,
                inc_std=self.inc_std,
                signed_distance_to_coast_km=signed_distance_to_coast_km,
                smooth_channels=self.smooth_channels,
                coast_aware=bool(self.cfg.get("coast_aware_smooth", True)),
                nearshore_km=float(self.cfg.get("smooth_nearshore_km", 50.0)),
                nearshore_weight=float(self.cfg.get("smooth_nearshore_weight", 0.3)),
            )

        loss_total = (
            loss_pred + self.lambda_obs * loss_obs + self.lambda_smooth * loss_smooth
        )

        logs = {
            "loss_total": loss_total.detach(),
            "loss_pred": loss_pred.detach(),
            "loss_obs": loss_obs.detach(),
            "loss_smooth": loss_smooth.detach(),
        }

        return loss_total, logs
