# train_bohai.py

import os
import time
import yaml
import json
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

# Import pyarrow before torch. In this DA environment, importing torch first can
# make pyarrow resolve an older system libstdc++, breaking parquet reads.
try:
    import pyarrow as _pyarrow  # noqa: F401
except ImportError:
    _pyarrow = None

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

import matplotlib.pyplot as plt

from dataset_bohai import build_dataloader
from loss_bohai import BohaiCompositeLoss

# ============================================================
# Basic utilities
# ============================================================


class AttrDict(dict):
    """
    Dictionary with attribute-style access.

    Example:
        cfg["lr"] and cfg.lr both work.
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def load_config(path: str) -> AttrDict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return AttrDict(cfg)


def configure_loss_from_type(cfg: AttrDict) -> AttrDict:
    """
    Translate a compact loss_type into the flags used by BohaiCompositeLoss.

    Supported loss_type values:
      E4 / pred:                  L_pred
      E5 / pred_obs:              L_pred + L_obs
      E6 / pred_smooth:           L_pred + L_smooth
      E7 / pred_obs_smooth:       L_pred + L_obs + L_smooth
      pmlf:                       L_inc + lambda_ms * multi-scale structure loss
      ae_recon:                   AE reconstruction with ae_loss_mode=mse or pmlf
      structured_ae:              StructuredSwinAE reconstruction + low/orth terms
    """
    loss_type = str(cfg.get("loss_type", "pred")).strip().lower()
    aliases = {
        "e4": "pred",
        "l_pred": "pred",
        "mse": "pred",
        "e5": "pred_obs",
        "l_pred+l_obs": "pred_obs",
        "pred+obs": "pred_obs",
        "e6": "pred_smooth",
        "l_pred+l_smooth": "pred_smooth",
        "pred+smooth": "pred_smooth",
        "e7": "pred_obs_smooth",
        "l_pred+l_obs+l_smooth": "pred_obs_smooth",
        "pred+obs+smooth": "pred_obs_smooth",
        "normalized_pmlf": "pmlf",
        "pmlf_style": "pmlf",
        "pgmi": "pmlf",
        "ae": "ae_recon",
        "autoencoder": "ae_recon",
        "reconstruction": "ae_recon",
        "structured": "structured_ae",
        "structured_autoencoder": "structured_ae",
    }
    loss_type = aliases.get(loss_type, loss_type)

    valid_loss_types = {
        "pred",
        "pred_obs",
        "pred_smooth",
        "pred_obs_smooth",
        "pmlf",
        "ae_recon",
        "structured_ae",
    }
    if loss_type not in valid_loss_types:
        raise ValueError(
            f"Unsupported loss_type={cfg.get('loss_type')!r}. "
            f"Use one of {sorted(valid_loss_types)} or E4/E5/E6/E7."
        )

    cfg["loss_type"] = loss_type
    cfg["use_obs_loss"] = loss_type in {"pred_obs", "pred_obs_smooth"}
    cfg["use_smooth_loss"] = loss_type in {"pred_smooth", "pred_obs_smooth"}

    if "lambda_obs" not in cfg:
        cfg["lambda_obs"] = 1.0 if cfg["use_obs_loss"] else 0.0
    if "lambda_smooth" not in cfg:
        cfg["lambda_smooth"] = 1.0 if cfg["use_smooth_loss"] else 0.0

    if loss_type == "pmlf":
        cfg["use_obs_loss"] = False
        cfg["use_smooth_loss"] = False
        cfg["lambda_obs"] = 0.0
        cfg["lambda_smooth"] = 0.0
        if "lambda_ms" not in cfg:
            cfg["lambda_ms"] = 0.1
    elif loss_type in {"ae_recon", "structured_ae"}:
        cfg["use_obs_loss"] = False
        cfg["use_smooth_loss"] = False
        cfg["lambda_obs"] = 0.0
        cfg["lambda_smooth"] = 0.0
        if "ae_loss_mode" not in cfg:
            legacy_rec = str(cfg.get("ae_rec_loss", "huber")).lower()
            legacy_lambda_ms = float(cfg.get("lambda_ms", 0.1))
            cfg["ae_loss_mode"] = (
                "mse" if legacy_rec == "mse" and legacy_lambda_ms <= 0 else "pmlf"
            )
        cfg["ae_loss_mode"] = str(cfg["ae_loss_mode"]).strip().lower()
        if cfg["ae_loss_mode"] in {"pure_mse", "mse_only"}:
            cfg["ae_loss_mode"] = "mse"
        if cfg["ae_loss_mode"] in {"normalized_pmlf", "full_pmlf"}:
            cfg["ae_loss_mode"] = "pmlf"
        if cfg["ae_loss_mode"] not in {"mse", "pmlf"}:
            raise ValueError("ae_loss_mode must be 'mse' or 'pmlf'.")

        if cfg["ae_loss_mode"] == "mse":
            cfg["ae_rec_loss"] = "mse"
            cfg["lambda_ms"] = 0.0
            cfg["ae_dynamic_alpha"] = False
        else:
            if "ae_rec_loss" not in cfg:
                cfg["ae_rec_loss"] = "huber"
            if "lambda_ms" not in cfg:
                cfg["lambda_ms"] = 0.1
            if "ae_dynamic_alpha" not in cfg:
                cfg["ae_dynamic_alpha"] = True

    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(
    device_arg: str = "auto", gpu_arg: int | None = None
) -> torch.device:
    """
    Resolve single-process training device.

    Examples:
      --device auto   -> cuda:0 if available else cpu
      --device cpu    -> CPU
      --device cuda   -> cuda:0
      --device cuda:1 -> GPU 1
      --gpu 1         -> cuda:1
    """
    if gpu_arg is not None:
        device_arg = f"cuda:{gpu_arg}"

    device_arg = str(device_arg).strip().lower()

    if device_arg == "auto":
        device_arg = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device_arg == "cuda":
        device_arg = "cuda:0"

    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device_arg}, but CUDA is not available.")

        if ":" in device_arg:
            try:
                index = int(device_arg.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid CUDA device: {device_arg}") from exc
        else:
            index = 0
            device_arg = "cuda:0"

        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested {device_arg}, but only {torch.cuda.device_count()} CUDA "
                f"device(s) are visible."
            )

        torch.cuda.set_device(index)

    return torch.device(device_arg)


def setup_distributed(device_arg: str = "auto", gpu_arg: int | None = None):
    """
    Supports both:
      1. single GPU / CPU:
         python train_bohai.py --config xxx.yaml --device cuda:1

      2. DDP:
         torchrun --nproc_per_node=2 train_bohai.py --config xxx.yaml
    """
    if "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        return True, rank, world_size, local_rank, device

    device = resolve_device(device_arg=device_arg, gpu_arg=gpu_arg)
    local_rank = (
        device.index if device.type == "cuda" and device.index is not None else 0
    )
    return False, 0, 1, local_rank, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model):
    if isinstance(model, DDP):
        return model.module
    return model


def count_parameters(model) -> int:
    model = unwrap_model(model)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def summarize_model_parameters(model) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Summarize current model parameters.

    Returns:
        summary:
            Whole-model parameter counts and approximate parameter memory.
        module_summary:
            Parameter counts grouped by top-level module prefix.
        parameter_summary:
            Per-parameter tensor shape/count and current value statistics.
    """
    model = unwrap_model(model)

    rows = []
    module_acc = {}
    total_params = 0
    trainable_params = 0
    total_bytes = 0

    for name, param in model.named_parameters():
        data = param.detach().float()
        numel = int(param.numel())
        bytes_ = int(param.numel() * param.element_size())
        trainable = bool(param.requires_grad)
        module = name.split(".", 1)[0] if "." in name else name

        total_params += numel
        total_bytes += bytes_
        if trainable:
            trainable_params += numel

        if module not in module_acc:
            module_acc[module] = {
                "module": module,
                "parameters": 0,
                "trainable_parameters": 0,
                "bytes": 0,
            }
        module_acc[module]["parameters"] += numel
        module_acc[module]["bytes"] += bytes_
        if trainable:
            module_acc[module]["trainable_parameters"] += numel

        rows.append(
            {
                "name": name,
                "module": module,
                "shape": "x".join(str(x) for x in param.shape),
                "dtype": str(param.dtype).replace("torch.", ""),
                "parameters": numel,
                "trainable": trainable,
                "mean": float(data.mean().item()) if numel > 0 else 0.0,
                "std": float(data.std(unbiased=False).item()) if numel > 1 else 0.0,
                "min": float(data.min().item()) if numel > 0 else 0.0,
                "max": float(data.max().item()) if numel > 0 else 0.0,
                "l2_norm": float(torch.linalg.vector_norm(data).item())
                if numel > 0
                else 0.0,
            }
        )

    summary = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "parameter_memory_bytes": total_bytes,
        "parameter_memory_mb": total_bytes / 1024**2,
    }

    module_summary = pd.DataFrame(module_acc.values())
    if not module_summary.empty:
        module_summary["frozen_parameters"] = (
            module_summary["parameters"] - module_summary["trainable_parameters"]
        )
        module_summary["parameter_memory_mb"] = module_summary["bytes"] / 1024**2
        module_summary = module_summary.sort_values(
            "parameters", ascending=False
        ).reset_index(drop=True)

    parameter_summary = pd.DataFrame(rows).sort_values(
        "parameters", ascending=False
    ).reset_index(drop=True)

    return summary, module_summary, parameter_summary


def save_parameter_report(model, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, module_summary, parameter_summary = summarize_model_parameters(model)

    (output_dir / "model_parameter_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    module_summary.to_csv(output_dir / "model_parameter_modules.csv", index=False)
    parameter_summary.to_csv(output_dir / "model_parameter_tensors.csv", index=False)

    return summary


def collect_gamma_attn(model) -> dict:
    model = unwrap_model(model)
    gammas = {}
    for name, param in model.named_parameters():
        if name.endswith("gamma_attn"):
            key = name.replace(".", "_")
            gammas[key] = float(param.detach().cpu().item())
    return gammas


def collect_hub_gates(model) -> dict:
    """Collect bounded, branch-specific Hub gates for V4 diagnostics."""
    model = unwrap_model(model)
    getter = getattr(model, "hub_gate_values", None)
    if getter is None:
        return {}
    values = getter(detach=True)
    return {
        str(name): float(value.detach().cpu().item())
        for name, value in values.items()
    }


def format_gamma_attn(gammas: dict) -> str:
    if not gammas:
        return "none"
    return ", ".join(f"{key}={value:.6g}" for key, value in gammas.items())


def reduce_mean(value: torch.Tensor, distributed: bool) -> torch.Tensor:
    """
    Average scalar tensor across all processes.
    """
    if distributed:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / dist.get_world_size()
    return value


def forward_model(model, inp: torch.Tensor, batch: dict, cfg) -> torch.Tensor:
    orth_cfg = dict(cfg.get("orth_loss", {}) or {})
    model_arch = str(cfg.get("model_arch", "")).lower()
    loss_type = str(cfg.get("loss_type", "")).lower()
    if model_arch in {
        "structured_swim_ae",
        "structed_swim_ae",
        "structured_swim_ae_refine",
        "structed_swim_ae_refine",
        "physp_ae",
        "physp_swin_ae",
        "physical_shared_private_ae",
        "physp_aev2",
        "physp_aev3",
        "physp_aev4",
        "structured_swim_ae_v2",
        "structured_swin_ae_v2",
    } or loss_type == "structured_ae":
        out = model(inp, return_parts=True)
        if isinstance(out, dict) and "full" not in out and "x_hat" in out:
            out["full"] = out["x_hat"]
        return out

    return_aux = (
        bool(cfg.get("use_branch_aux_loss", False))
        or float(cfg.get("lambda_branch", 0.0)) > 0
        or bool(orth_cfg.get("enabled", False))
        or float(orth_cfg.get("weight", cfg.get("lambda_orth", 0.0))) > 0
    )
    if bool(cfg.get("use_obs_tokens", False)):
        return model(
            inp,
            obs_feat=batch.get("obs_feat"),
            obs_yx=batch.get("obs_yx"),
            obs_conf=batch.get("obs_conf"),
            obs_mask=batch.get("obs_mask"),
            neighbor_idx=batch.get("neighbor_idx"),
            neighbor_mask=batch.get("neighbor_mask"),
            return_aux=return_aux,
        )
    try:
        return model(inp, return_aux=return_aux)
    except TypeError:
        return model(inp)


# ============================================================
# Metrics
# These metrics are computed in normalized increment space.
# Physical-space metrics should be computed in evaluate_bohai.py.
# ============================================================


def torch_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def torch_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def torch_corr(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Overall Pearson correlation for tensors.
    """
    x = pred.reshape(-1)
    y = target.reshape(-1)

    x = x - torch.mean(x)
    y = y - torch.mean(y)

    denom = torch.sqrt(torch.sum(x**2) * torch.sum(y**2))
    return torch.sum(x * y) / (denom + eps)


def channelwise_metrics(pred: torch.Tensor, target: torch.Tensor, channel_names):
    """
    Compute RMSE / MAE / CORR for each output channel plus mean metric.

    pred, target:
        [B, C, H, W]
    """
    logs = {}

    logs["rmse_mean"] = torch_rmse(pred, target).detach()
    logs["mae_mean"] = torch_mae(pred, target).detach()
    logs["corr_mean"] = torch_corr(pred, target).detach()

    for ci, name in enumerate(channel_names):
        p = pred[:, ci : ci + 1]
        t = target[:, ci : ci + 1]

        logs[f"rmse_{name}"] = torch_rmse(p, t).detach()
        logs[f"mae_{name}"] = torch_mae(p, t).detach()
        logs[f"corr_{name}"] = torch_corr(p, t).detach()

    return logs


# ============================================================
# Plot utilities
# ============================================================


def save_history_csv(history, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    df.to_csv(output_csv, index=False)


def plot_loss_curve(history_csv: Path, output_png: Path):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(history_csv)

    plt.figure(figsize=(8, 5))

    if "train_loss" in df.columns:
        plt.plot(df["epoch"], df["train_loss"], label="train_loss")

    if "valid_loss" in df.columns:
        valid_df = df.dropna(subset=["valid_loss"])
        if len(valid_df) > 0:
            plt.plot(valid_df["epoch"], valid_df["valid_loss"], label="valid_loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Valid Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def plot_metric_curve(history_csv: Path, output_dir: Path, metric_names):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(history_csv)

    for metric in metric_names:
        if metric not in df.columns:
            continue

        valid_df = df.dropna(subset=[metric])
        if len(valid_df) == 0:
            continue

        plt.figure(figsize=(8, 5))
        plt.plot(valid_df["epoch"], valid_df[metric], label=metric)

        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(metric)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}.png", dpi=200)
        plt.close()


def plot_channel_metric_bars(
    history_csv: Path, output_dir: Path, prefix: str, epoch=None
):
    """
    Draw bar plot for channel-wise metrics at the last epoch by default.

    Example:
        prefix = "valid_rmse"
        columns:
            valid_rmse_inc_t2m
            valid_rmse_inc_u10
            ...
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(history_csv)

    if len(df) == 0:
        return

    if epoch is None:
        row = df.iloc[-1]
        epoch = int(row["epoch"])
    else:
        rows = df[df["epoch"] == epoch]
        if len(rows) == 0:
            return
        row = rows.iloc[0]

    cols = [
        c for c in df.columns if c.startswith(prefix + "_") and not c.endswith("_mean")
    ]

    if len(cols) == 0:
        return

    names = [c.replace(prefix + "_", "") for c in cols]
    values = [row[c] for c in cols]

    plt.figure(figsize=(9, 5))
    plt.bar(names, values)
    plt.xlabel("Variable")
    plt.ylabel(prefix)
    plt.title(f"{prefix} at epoch {epoch}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_bar_epoch_{epoch}.png", dpi=200)
    plt.close()


def update_plots(history_csv: Path, fig_dir: Path):
    plot_loss_curve(
        history_csv=history_csv,
        output_png=fig_dir / "loss_curve.png",
    )

    plot_metric_curve(
        history_csv=history_csv,
        output_dir=fig_dir,
        metric_names=[
            "valid_rmse_mean",
            "valid_mae_mean",
            "valid_corr_mean",
        ],
    )

    plot_channel_metric_bars(
        history_csv=history_csv,
        output_dir=fig_dir,
        prefix="valid_rmse",
    )
    plot_channel_metric_bars(
        history_csv=history_csv,
        output_dir=fig_dir,
        prefix="valid_mae",
    )
    plot_channel_metric_bars(
        history_csv=history_csv,
        output_dir=fig_dir,
        prefix="valid_corr",
    )


# ============================================================
# Checkpoint utilities
# ============================================================


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_valid_loss: float,
    cfg,
    distributed: bool,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    if distributed:
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    ckpt = {
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_valid_loss": best_valid_loss,
        "config": dict(cfg),
    }

    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
    distributed: bool,
):
    ckpt = torch.load(path, map_location=device)

    if distributed:
        model.module.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    start_epoch = int(ckpt["epoch"]) + 1
    best_valid_loss = float(ckpt.get("best_valid_loss", float("inf")))

    return start_epoch, best_valid_loss


def load_model_weights(path: str, model, device, distributed: bool):
    """
    Load model weights only for fine-tuning.

    This intentionally does not restore optimizer, scheduler, scaler, epoch, or
    best metric state. Use --resume for true interrupted-run continuation.
    """
    ckpt = torch.load(path, map_location=device)
    model_state = ckpt.get("model_state", ckpt)

    if distributed:
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

    return ckpt


# ============================================================
# Train / Validate
# ============================================================


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    cfg,
    device,
    epoch: int,
    distributed: bool,
    is_main: bool = True,
):
    model.train()

    if distributed and loader.sampler is not None:
        loader.sampler.set_epoch(epoch)

    total_loss = torch.tensor(0.0, device=device)
    total_steps = torch.tensor(0.0, device=device)
    loss_log_sums = {}

    start = time.time()

    progress = tqdm(
        loader,
        desc=f"Train Epoch {epoch}",
        disable=not is_main,
        dynamic_ncols=True,
    )

    for batch in progress:
        inp = batch["input"].to(device, dtype=torch.float32, non_blocking=True)
        target = batch["target"].to(device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=bool(cfg.get("enable_amp", True))):
            pred = forward_model(model, inp, batch, cfg)
            loss, loss_logs = criterion(pred, target, batch)

        if bool(cfg.get("enable_amp", True)):
            scaler.scale(loss).backward()

            if cfg.get("grad_clip", None) is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(cfg["grad_clip"]),
                )

            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()

            if cfg.get("grad_clip", None) is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(cfg["grad_clip"]),
                )

            optimizer.step()

        total_loss += loss.detach()
        total_steps += 1.0
        for key, value in loss_logs.items():
            if key not in loss_log_sums:
                loss_log_sums[key] = torch.tensor(0.0, device=device)
            loss_log_sums[key] += value.to(device)

        if is_main:
            progress.set_postfix(
                {
                    "loss": f"{loss.item():.5f}",
                    "pred": f"{loss_logs['loss_pred'].item():.5f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

    total_loss = reduce_mean(total_loss, distributed)
    total_steps = reduce_mean(total_steps, distributed)
    for key in loss_log_sums:
        loss_log_sums[key] = reduce_mean(loss_log_sums[key], distributed)

    avg_loss = (total_loss / total_steps).item()
    avg_loss_logs = {
        f"train_{key}": (value / total_steps).item()
        for key, value in loss_log_sums.items()
    }
    elapsed = time.time() - start

    return avg_loss, avg_loss_logs, elapsed


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    cfg,
    device,
    distributed: bool,
    is_main: bool = True,
):
    model.eval()

    target_names = list(cfg["target_vars"])

    total_loss = torch.tensor(0.0, device=device)
    total_steps = torch.tensor(0.0, device=device)
    loss_log_sums = {}

    metric_sums = {}
    metric_steps = torch.tensor(0.0, device=device)

    progress = tqdm(
        loader,
        desc="Valid",
        disable=not is_main,
        dynamic_ncols=True,
    )

    for batch in progress:
        inp = batch["input"].to(device, dtype=torch.float32, non_blocking=True)
        target = batch["target"].to(device, dtype=torch.float32, non_blocking=True)

        with autocast(enabled=bool(cfg.get("enable_amp", True))):
            pred = forward_model(model, inp, batch, cfg)
            pred_full = pred["full"] if isinstance(pred, dict) else pred
            pred_mse = F.mse_loss(pred_full, target)
            total_objective, loss_logs = criterion(pred, target, batch)

        total_loss += pred_mse.detach()
        total_steps += 1.0
        for key, value in loss_logs.items():
            if key not in loss_log_sums:
                loss_log_sums[key] = torch.tensor(0.0, device=device)
            loss_log_sums[key] += value.to(device)

        batch_metrics = channelwise_metrics(
            pred=pred_full.float(),
            target=target.float(),
            channel_names=target_names,
        )

        for key, value in batch_metrics.items():
            if key not in metric_sums:
                metric_sums[key] = torch.tensor(0.0, device=device)
            metric_sums[key] += value.to(device)

        metric_steps += 1.0

        if is_main:
            progress.set_postfix(
                {
                    "valid_loss": f"{pred_mse.item():.5f}",
                    "valid_total": f"{total_objective.item():.5f}",
                }
            )

    total_loss = reduce_mean(total_loss, distributed)
    total_steps = reduce_mean(total_steps, distributed)
    for key in loss_log_sums:
        loss_log_sums[key] = reduce_mean(loss_log_sums[key], distributed)

    valid_loss = total_loss / total_steps

    for key in metric_sums:
        metric_sums[key] = reduce_mean(metric_sums[key], distributed)

    metric_steps = reduce_mean(metric_steps, distributed)

    logs = {
        "valid_loss": valid_loss.item(),
    }

    for key, value in loss_log_sums.items():
        logs[f"valid_{key}"] = (value / total_steps).item()

    for key, value in metric_sums.items():
        logs[f"valid_{key}"] = (value / metric_steps).item()

    return logs


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument(
        "--warm-start",
        default=None,
        type=str,
        help=(
            "Load model weights only and start a fresh training run. "
            "Use this for fine-tuning with a new loss/config."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        type=str,
        help=(
            "Device for single-process training: auto, cpu, cuda, cuda:0, cuda:1, ..."
            " Ignored when launched by torchrun/DDP."
        ),
    )
    parser.add_argument(
        "--gpu",
        default=None,
        type=int,
        help="Shortcut for --device cuda:<id> in single-process training.",
    )
    args = parser.parse_args()

    if args.resume is not None and args.warm_start is not None:
        raise ValueError("--resume and --warm-start are mutually exclusive.")

    cfg = configure_loss_from_type(load_config(args.config))

    distributed, rank, world_size, local_rank, device = setup_distributed(
        device_arg=args.device,
        gpu_arg=args.gpu,
    )
    is_main = is_main_process(rank)

    seed = int(cfg.get("seed", 42))
    set_seed(seed + rank)

    exp_dir = Path(cfg["exp_dir"])
    ckpt_dir = exp_dir / "checkpoints"
    log_dir = exp_dir / "logs"
    fig_dir = exp_dir / "figures"

    if is_main:
        exp_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Save config copy for reproducibility.
        with open(exp_dir / "config_used.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(cfg), f, sort_keys=False, allow_unicode=True)

    if distributed:
        dist.barrier()

    if is_main:
        print("=" * 80)
        print("Bohai-Yellow Sea AI Data Assimilation Training")
        print(f"Distributed: {distributed}")
        print(f"World size: {world_size}")
        print(f"Device: {device}")
        print(f"Experiment directory: {exp_dir}")
        print("=" * 80)

    # ----------------------------
    # Data
    # ----------------------------
    train_loader, train_dataset, train_sampler = build_dataloader(
        cfg,
        split="train",
        distributed=distributed,
    )

    valid_loader, valid_dataset, valid_sampler = build_dataloader(
        cfg,
        split="valid",
        distributed=distributed,
    )

    criterion = BohaiCompositeLoss(
        cfg=cfg,
        norm_stats=train_dataset.norm_stats,
        device=device,
    )

    # ----------------------------
    # Model
    # ----------------------------
    model_arch = str(cfg.get("model_arch", "encdec")).lower()
    if model_arch == "swim_ir_v2":
        from models.SwimIRv2 import SwimIRv2

        model = SwimIRv2(cfg).to(device)
    elif model_arch == "swim_ir_v3":
        from models.SwimIRv3 import SwimIRv3

        model = SwimIRv3(cfg).to(device)
    elif model_arch == "hat":
        from models.hat_bohai import HATBohai

        model = HATBohai(cfg).to(device)
    elif model_arch == "swim_ir_ae":
        from models.SwimIR_AE import SwimIR_AE

        model = SwimIR_AE(cfg).to(device)
    elif model_arch in {"shared_private_swim_ir_ae", "shared_private_swimir_ae"}:
        from models.SharedPrivateSwimIR_AE import SharedPrivateSwimIR_AE

        model = SharedPrivateSwimIR_AE(cfg).to(device)
    elif model_arch in {"structured_swim_ae", "structed_swim_ae"}:
        from models.StructedSwimAE import StructuredSwinAE

        model = StructuredSwinAE(cfg).to(device)
    elif model_arch in {"structured_swim_ae_refine", "structed_swim_ae_refine"}:
        from models.ocs_lda_blocks import StructuredSwinAERefine

        model = StructuredSwinAERefine(cfg).to(device)
    elif model_arch in {"physp_ae", "physp_swin_ae", "physical_shared_private_ae"}:
        from models.PhySP_AE import PhySP_AE

        model = PhySP_AE(cfg).to(device)
    elif model_arch == "physp_aev2":
        from models.PhySP_AEv2 import PhySP_AEv2

        model = PhySP_AEv2(cfg).to(device)
    elif model_arch == "physp_aev3":
        from models.ocs_lda import PhySP_AEv3

        model = PhySP_AEv3(cfg).to(device)
    elif model_arch == "physp_aev4":
        from models.ocs_lda import PhySP_AEv4

        model = PhySP_AEv4(cfg).to(device)
    elif model_arch in {"structured_swim_ae_v2", "structured_swin_ae_v2"}:
        from models.StructuredSwinAEv2 import StructuredSwinAEv2

        model = StructuredSwinAEv2(cfg).to(device)
    elif model_arch == "lda_vit_ae":
        from models.LDA_ViT_AE import LDA_ViT_AE

        model = LDA_ViT_AE(cfg).to(device)
    elif model_arch == "lda141_ae":
        from models.LDA141_AE import LDA141_AE

        model = LDA141_AE(cfg).to(device)
    else:
        from models.encdec import EncDec

        model = EncDec(cfg).to(device)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    if is_main:
        print(model)
        param_summary = save_parameter_report(model, log_dir)
        print(f"Total parameters: {param_summary['total_parameters']:,}")
        print(f"Trainable parameters: {param_summary['trainable_parameters']:,}")
        print(f"Frozen parameters: {param_summary['frozen_parameters']:,}")
        print(
            f"Parameter memory: {param_summary['parameter_memory_mb']:.2f} MB"
        )
        print(f"Parameter report: {log_dir}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Valid samples: {len(valid_dataset)}")
        print(f"Input channels: {cfg['in_chans']}")
        print(f"Output channels: {cfg['out_chans']}")
        print(f"Target vars: {cfg['target_vars']}")
        print(f"Obs fusion gamma_attn: {format_gamma_attn(collect_gamma_attn(model))}")
        initial_hub_gates = collect_hub_gates(model)
        print(
            "Hub gates: "
            + (
                ", ".join(
                    f"{name}={value:.6g}"
                    for name, value in initial_hub_gates.items()
                )
                if initial_hub_gates
                else "none"
            )
        )
        print(
            "Loss: "
            f"{cfg['loss_type']} "
            f"(lambda_obs={float(cfg.get('lambda_obs', 0.0))}, "
            f"lambda_smooth={float(cfg.get('lambda_smooth', 0.0))}, "
            f"lambda_ms={float(cfg.get('lambda_ms', 0.0))})"
        )

    # ----------------------------
    # Optimizer
    # ----------------------------
    optimizer_name = cfg.get("optimizer", "AdamW")

    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
        )
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg["lr"]),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # ----------------------------
    # Scheduler
    # ----------------------------
    scheduler_name = cfg.get("scheduler", "CosineAnnealingLR")

    if scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg["max_epochs"]),
            eta_min=float(cfg.get("min_lr", 1e-6)),
        )
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(cfg.get("lr_reduce_factor", 0.5)),
            patience=int(cfg.get("lr_patience", 10)),
        )
    elif scheduler_name in ["None", None]:
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    # ----------------------------
    # AMP scaler
    # ----------------------------
    scaler = GradScaler(enabled=bool(cfg.get("enable_amp", True)))

    # ----------------------------
    # Resume
    # ----------------------------
    start_epoch = 0
    best_valid_loss = float("inf")
    history = []

    history_csv = log_dir / "history.csv"

    if args.warm_start is not None:
        load_model_weights(
            path=args.warm_start,
            model=model,
            device=device,
            distributed=distributed,
        )
        if is_main:
            print(f"Warm-started model weights from: {args.warm_start}")
            print("Optimizer, scheduler, scaler, epoch, and best metric were reset.")

    if args.resume is not None:
        start_epoch, best_valid_loss = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            distributed=distributed,
        )

        if history_csv.exists():
            history_df = pd.read_csv(history_csv)
            history = history_df.to_dict("records")

        if is_main:
            print(f"Resumed from: {args.resume}")
            print(f"Start epoch: {start_epoch}")
            print(f"Best valid loss: {best_valid_loss:.6f}")

    # ----------------------------
    # Train loop
    # ----------------------------
    max_epochs = int(cfg["max_epochs"])
    valid_frequency = int(cfg.get("valid_frequency", 1))
    save_frequency = int(cfg.get("save_frequency", 5))
    early_stopping_patience = cfg.get("early_stopping_patience", None)
    early_stopping_patience = (
        int(early_stopping_patience)
        if early_stopping_patience is not None
        else None
    )
    early_stopping_min_delta = float(cfg.get("early_stopping_min_delta", 0.0))
    early_stopping_bad_epochs = 0
    early_stopping_triggered = False
    last_epoch_completed = start_epoch - 1

    if is_main and early_stopping_patience is not None:
        print(
            "Early stopping enabled: "
            f"patience={early_stopping_patience}, "
            f"min_delta={early_stopping_min_delta:g}"
        )

    for epoch in range(start_epoch, max_epochs):
        train_loss, train_logs, train_time = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            cfg=cfg,
            device=device,
            epoch=epoch,
            distributed=distributed,
            is_main=is_main,
        )
        last_epoch_completed = epoch

        do_valid = (epoch % valid_frequency == 0) or (epoch == max_epochs - 1)

        if do_valid:
            valid_logs = validate(
                model=model,
                loader=valid_loader,
                criterion=criterion,
                cfg=cfg,
                device=device,
                distributed=distributed,
                is_main=is_main,
            )
            valid_loss = valid_logs["valid_loss"]
        else:
            valid_logs = {}
            valid_loss = np.nan

        # Scheduler step
        if scheduler is not None:
            if scheduler_name == "ReduceLROnPlateau":
                if not np.isnan(valid_loss):
                    scheduler.step(valid_loss)
            else:
                scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        # ----------------------------
        # Logging / plotting / saving
        # ----------------------------
        if is_main:
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "lr": lr,
                "train_time_sec": train_time,
            }

            record.update(train_logs)
            gamma_logs = collect_gamma_attn(model)
            for key, value in gamma_logs.items():
                record[f"gamma_attn_{key}"] = value

            hub_gate_logs = collect_hub_gates(model)
            for key, value in hub_gate_logs.items():
                record[f"hub_gate_{key}"] = value

            for key, value in valid_logs.items():
                if key != "valid_loss":
                    record[key] = value

            history.append(record)

            save_history_csv(history, history_csv)
            update_plots(history_csv, fig_dir)

            msg = f"Epoch {epoch:04d} | " f"train_loss={train_loss:.6f} | "

            if not np.isnan(valid_loss):
                msg += (
                    f"valid_loss={valid_loss:.6f} | "
                    f"valid_rmse={valid_logs.get('valid_rmse_mean', np.nan):.6f} | "
                    f"valid_mae={valid_logs.get('valid_mae_mean', np.nan):.6f} | "
                    f"valid_corr={valid_logs.get('valid_corr_mean', np.nan):.6f} | "
                )

            gamma_msg = format_gamma_attn(gamma_logs)
            if gamma_msg != "none":
                msg += f"gamma_attn=[{gamma_msg}] | "

            if hub_gate_logs:
                hub_gate_msg = ", ".join(
                    f"{name}={value:.6g}"
                    for name, value in hub_gate_logs.items()
                )
                msg += f"hub_gates=[{hub_gate_msg}] | "

            msg += f"lr={lr:.3e} | " f"time={train_time:.1f}s"

            print(msg)

            # Save latest checkpoint
            if epoch % save_frequency == 0:
                save_checkpoint(
                    path=ckpt_dir / "latest.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_valid_loss=best_valid_loss,
                    cfg=cfg,
                    distributed=distributed,
                )

            # Save best checkpoint
            improved = (
                not np.isnan(valid_loss)
                and valid_loss < best_valid_loss - early_stopping_min_delta
            )

            if improved:
                best_valid_loss = float(valid_loss)
                early_stopping_bad_epochs = 0

                save_checkpoint(
                    path=ckpt_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_valid_loss=best_valid_loss,
                    cfg=cfg,
                    distributed=distributed,
                )

                print(f"Saved best checkpoint: best_valid_loss={best_valid_loss:.6f}")
            elif do_valid and not np.isnan(valid_loss):
                early_stopping_bad_epochs += 1

            if (
                early_stopping_patience is not None
                and early_stopping_bad_epochs >= early_stopping_patience
            ):
                early_stopping_triggered = True
                print(
                    "Early stopping triggered: "
                    f"no validation improvement for {early_stopping_bad_epochs} "
                    f"validation checks. Best valid loss={best_valid_loss:.6f}"
                )

        if distributed:
            stop_tensor = torch.tensor(
                int(early_stopping_triggered),
                device=device,
                dtype=torch.int32,
            )
            dist.broadcast(stop_tensor, src=0)
            early_stopping_triggered = bool(stop_tensor.item())

        if early_stopping_triggered:
            break

    if is_main:
        # Always save final checkpoint
        save_checkpoint(
            path=ckpt_dir / "final.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=max(last_epoch_completed, start_epoch),
            best_valid_loss=best_valid_loss,
            cfg=cfg,
            distributed=distributed,
        )

        print("=" * 80)
        print("Training finished.")
        if early_stopping_triggered:
            print(f"Stopped early at epoch: {last_epoch_completed}")
        print(f"Best valid loss: {best_valid_loss:.6f}")
        print(f"History: {history_csv}")
        print(f"Figures: {fig_dir}")
        print(f"Checkpoints: {ckpt_dir}")
        print("=" * 80)

    cleanup_distributed()


if __name__ == "__main__":
    main()
