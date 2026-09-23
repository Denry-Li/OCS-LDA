"""Unified deterministic 6-hour cycling OSSE for Bohai 3DVar methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecast_inference import load_config as load_forecast_config
from forecast_inference import load_model as load_forecast_model
from forecast_time_features import build_time_features
from latent_3dvar_bohai import (
    branch_background_loss,
    branch_increment_logs,
    branch_slices_from_model,
    decode_det_with_size,
    encode_det_with_size,
    latent_background_loss,
    load_config as load_ae_config,
    load_latent_bz,
    load_model as load_ae_model,
    normalize_active_branches,
    pack_active_latent,
)
from utils.point_observation_operator import (
    GdasOssePointObsLoader,
    normalization_vectors,
    point_observation_loss,
    sample_point_observations,
)


METHODS = ["PhySP_DA"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stop-after", type=int, default=-1)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resolve_device(name):
    name = str(name).lower()
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    if name == "cuda":
        name = "cuda:0"
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {name}, but CUDA is unavailable")
        torch.cuda.set_device(int(name.split(":", 1)[1]))
    return torch.device(name)


def checkpoint_save_kind(completed_cycles, stop_after, total_cycles, mode, interval):
    """Return ``per_cycle``, ``latest``, ``final``, or ``None`` for this cycle.

    ``rolling`` mode overwrites one latest checkpoint at the configured
    interval.  A deliberately shortened run also saves its last state, even
    when that cycle is not an interval boundary.  A fully completed run saves
    only its numbered final checkpoint.
    """
    completed_cycles = int(completed_cycles)
    stop_after = int(stop_after)
    total_cycles = int(total_cycles)
    interval = int(interval)
    mode = str(mode).lower()
    if interval <= 0:
        raise ValueError(f"checkpoint_interval_cycles must be positive, got {interval}")
    if mode not in {"per_cycle", "rolling"}:
        raise ValueError(
            f"checkpoint_mode must be 'per_cycle' or 'rolling', got {mode!r}"
        )
    if mode == "per_cycle":
        return "per_cycle"
    if completed_cycles == total_cycles:
        return "final"
    if completed_cycles % interval == 0 or completed_cycles == stop_after:
        return "latest"
    return None


def select_latent_bz_mode(background_error, mode, name, branch_names=None):
    """Select the configured latent covariance approximation after loading/scaling."""
    mode = str(mode).lower()
    if mode == "channel_covariance":
        return background_error
    if mode == "semantic_branch_block":
        if not isinstance(background_error, dict) or background_error.get("type") != "latent_channel_cov":
            raise TypeError(f"{name} semantic branch-block B_z requires a latent_channel_cov file.")
        branch_names = tuple(
            branch_names or ("shared", "wind", "thermo_moist", "pressure")
        )
        items = background_error.get("items", {})
        missing = [branch for branch in branch_names if branch not in items]
        if missing:
            raise KeyError(f"{name} B_z is missing semantic branch items: {missing}")
        branch_sizes = []
        for branch in branch_names:
            item = items[branch]
            covariance = item.get("cov_loaded")
            cholesky = item.get("cholesky")
            if cholesky is None:
                raise KeyError(f"{name} semantic branch {branch!r} lacks cholesky.")
            if cholesky.ndim != 2 or cholesky.shape[0] != cholesky.shape[1]:
                raise ValueError(f"{name} branch {branch!r} Cholesky must be square.")
            if covariance is not None:
                if covariance.shape != cholesky.shape:
                    raise ValueError(
                        f"{name} branch {branch!r} Cholesky shape differs from covariance."
                    )
                reconstructed = cholesky @ cholesky.transpose(-1, -2)
                if not torch.allclose(reconstructed, covariance, rtol=1e-4, atol=1e-6):
                    raise ValueError(
                        f"{name} semantic branch {branch!r} Cholesky does not reconstruct covariance."
                    )
            if not bool(torch.isfinite(cholesky).all()):
                raise ValueError(f"{name} branch {branch!r} Cholesky must be finite.")
            branch_sizes.append(int(cholesky.shape[0]))
        selected = dict(background_error)
        # Preserve independent blocks.  Never rebuild a dense block-diagonal
        # matrix: the optimizer evaluates one quadratic form per branch.
        selected["items"] = {branch: items[branch] for branch in branch_names}
        selected["covariance_mode"] = "semantic_branch_block"
        selected["semantic_branch_names"] = list(branch_names)
        selected["semantic_branch_sizes"] = branch_sizes
        return selected
    if mode == "elementwise_diagonal":
        if not isinstance(background_error, dict) or background_error.get("type") != "latent_channel_cov":
            raise TypeError(f"{name} elementwise diagonal B_z requires a latent_channel_cov file.")
        item = background_error.get("items", {}).get("latent")
        if item is None or "diag_var_spatial" not in item:
            raise KeyError(f"{name} B_z does not contain items['latent']['diag_var_spatial'].")
        diagonal = item["diag_var_spatial"]
        if diagonal.ndim != 3:
            raise ValueError(f"{name} diagonal B_z must have shape [C,H,W], got {tuple(diagonal.shape)}")
        if not bool(torch.isfinite(diagonal).all()) or bool((diagonal <= 0).any()):
            raise ValueError(f"{name} diagonal B_z must be positive and finite.")
        return diagonal
    raise ValueError(f"Unsupported {name} B_z mode: {mode}")


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temp)
    os.replace(temp, path)


def atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def collect_physp_hub_gates(model):
    """Return frozen Hub gate values for experiment provenance."""
    if hasattr(model, "hub_gate_values"):
        values = model.hub_gate_values(detach=True)
        return {
            str(name): float(value.detach().cpu().item())
            for name, value in values.items()
        }
    branch_latent = getattr(model, "branch_latent", None)
    hub = getattr(branch_latent, "hub", None)
    beta = getattr(hub, "beta", None)
    if beta is not None:
        return {"global_beta": float(beta.detach().cpu().item())}
    return None


def standardize(array, variables, stats):
    output = np.empty_like(array, dtype=np.float32)
    for channel, name in enumerate(variables):
        mean = float(stats[name]["mean"])
        std = float(stats[name]["std"]) + 1e-6
        output[channel] = (array[channel] - mean) / std
    return output


def destandardize_tensor(tensor, variables, stats):
    mean = torch.tensor(
        [float(stats[name]["mean"]) for name in variables],
        device=tensor.device,
        dtype=tensor.dtype,
    ).view(1, -1, 1, 1)
    std = torch.tensor(
        [float(stats[name]["std"]) + 1e-6 for name in variables],
        device=tensor.device,
        dtype=tensor.dtype,
    ).view(1, -1, 1, 1)
    return tensor * std + mean


def read_physical_state(dataset, time_value, variables):
    time64 = np.datetime64(pd.Timestamp(time_value).to_datetime64(), "ns")
    indices = np.flatnonzero(dataset.time.values.astype("datetime64[ns]") == time64)
    if indices.size != 1:
        raise KeyError(f"Expected one state at {time_value}, found {indices.size}")
    return np.stack(
        [dataset[name].isel(time=int(indices[0])).values.astype(np.float32) for name in variables],
        axis=0,
    )


def forecast_one_step(model, previous, current, valid_time, lat, lon, mode):
    feature = build_time_features(valid_time, lat, lon, mode=mode)
    feature = torch.from_numpy(feature).to(current.device, dtype=current.dtype).unsqueeze(0)
    with torch.no_grad():
        return model(previous, current, feature).detach()


def field_metrics(state_std, truth_std, variables, stats):
    state = destandardize_tensor(state_std, variables, stats)[0].double().cpu()
    truth = destandardize_tensor(truth_std, variables, stats)[0].double().cpu()
    rows = {}
    for channel, name in enumerate(variables):
        pred = state[channel].reshape(-1)
        target = truth[channel].reshape(-1)
        diff = pred - target
        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        denom = torch.sqrt(pred_centered.square().sum() * target_centered.square().sum())
        corr = (pred_centered * target_centered).sum() / denom.clamp_min(1e-12)
        rows[name] = {
            "rmse": float(torch.sqrt(diff.square().mean())),
            "mae": float(diff.abs().mean()),
            "bias": float(diff.mean()),
            "corr": float(corr),
            "minimum": float(pred.min()),
            "maximum": float(pred.max()),
        }
    return rows


def observation_metrics(state_std, observation, variables, stats):
    mean, std = normalization_vectors(stats, variables, state_std.device, state_std.dtype)
    pred_std = sample_point_observations(state_std[0], observation)
    pred = pred_std * std[observation.channel_index] + mean[observation.channel_index]
    innovation = pred - observation.value
    result = {}
    for channel, name in enumerate(variables):
        mask = observation.channel_index == channel
        count = int(mask.sum())
        if count == 0:
            result[name] = {"count": 0, "rmse": None, "bias": None, "chi2": None}
            continue
        diff = innovation[mask]
        variance = observation.error_variance[mask]
        result[name] = {
            "count": count,
            "rmse": float(torch.sqrt(diff.square().mean()).detach().cpu()),
            "bias": float(diff.mean().detach().cpu()),
            "chi2": float((diff.square() / variance).mean().detach().cpu()),
        }
    return result


def evaluate_latent_cost(
    model,
    z,
    z_b,
    output_size,
    observations,
    variables,
    stats,
    bz,
    active_z=None,
    branch_slices=None,
    background=None,
    manifold_background=None,
    analysis_decode_mode="absolute_latent_state",
):
    if active_z is None:
        background_cost = latent_background_loss(z, z_b, bz, reduction="sum")
    else:
        background_cost = branch_background_loss(
            active_z,
            z_b,
            branch_slices,
            bz,
            reduction="sum",
        )
    decoded = decode_det_with_size(model, z, output_size)
    if analysis_decode_mode == "absolute_latent_state":
        analysis = decoded
    elif analysis_decode_mode == "background_anchored_increment":
        if background is None or manifold_background is None:
            raise ValueError(
                "background_anchored_increment requires background and "
                "manifold_background"
            )
        # Exact incremental anchoring: z == z_b implies analysis == background.
        # The cached manifold_background is detached, while gradients propagate
        # through decoded to the optimized latent increment.
        analysis = background + (decoded - manifold_background)
    else:
        raise ValueError(
            f"Unknown analysis_decode_mode={analysis_decode_mode!r}; expected "
            "'absolute_latent_state' or 'background_anchored_increment'."
        )
    observation_cost, diagnostics = point_observation_loss(
        analysis[:, : len(variables)],
        observations,
        variables,
        norm_stats=stats,
        reduction="sum",
    )
    return background_cost, observation_cost, analysis, diagnostics


def optimize_latent(
    model,
    background,
    observations,
    variables,
    stats,
    bz,
    lr,
    max_steps,
    grad_clip,
    patience,
    tolerance,
    active_branches=None,
    analysis_decode_mode="absolute_latent_state",
):
    started_at = time.perf_counter()
    if background.is_cuda:
        torch.cuda.reset_peak_memory_stats(background.device)
    with torch.no_grad():
        z_b, output_size = encode_det_with_size(model, background)
        manifold = decode_det_with_size(model, z_b, output_size).detach()
    if analysis_decode_mode not in {
        "absolute_latent_state",
        "background_anchored_increment",
    }:
        raise ValueError(
            f"Unknown analysis_decode_mode={analysis_decode_mode!r}; expected "
            "'absolute_latent_state' or 'background_anchored_increment'."
        )
    ae_background_delta_rms = float(
        torch.sqrt((manifold - background).square().mean()).detach().cpu()
    )
    slices = branch_slices_from_model(model)
    requested_branches = normalize_active_branches(active_branches)
    semantic_bz = (
        isinstance(bz, dict)
        and bz.get("covariance_mode") == "semantic_branch_block"
    )
    if semantic_bz and requested_branches is None:
        requested_branches = list(bz["semantic_branch_names"])
    if requested_branches is not None:
        if slices is None:
            raise ValueError("Active latent branches require a branch-structured model.")
        unknown = sorted(set(requested_branches) - set(slices))
        if unknown:
            raise ValueError(
                f"Unknown active branches {unknown}; available={list(slices)}"
            )
        if semantic_bz:
            missing_bz = sorted(set(requested_branches) - set(bz["items"]))
            if missing_bz:
                raise KeyError(
                    f"Semantic B_z lacks active branches {missing_bz}; "
                    f"loaded={list(bz['items'])}"
                )
        active_z = {
            name: z_b[:, slices[name]].detach().clone().requires_grad_(True)
            for name in requested_branches
        }
        optimizer_parameters = list(active_z.values())
        optimizer = torch.optim.Adam(optimizer_parameters, lr=float(lr))
        z = pack_active_latent(z_b.detach(), slices, active_z)
    else:
        active_z = None
        z = z_b.detach().clone().requires_grad_(True)
        optimizer_parameters = [z]
        optimizer = torch.optim.Adam(optimizer_parameters, lr=float(lr))
    best = None
    patience_best_loss = None
    no_improvement_steps = 0
    early_stopping_min_delta = None
    stop_reason = "max_steps"
    iterations = 0
    last_grad_norm = None
    initial = None

    for step in range(int(max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        if active_z is not None:
            z = pack_active_latent(z_b.detach(), slices, active_z)
        loss_b, loss_o, decoded, diagnostics = evaluate_latent_cost(
            model,
            z,
            z_b.detach(),
            output_size,
            observations,
            variables,
            stats,
            bz,
            active_z=active_z,
            branch_slices=slices,
            background=background,
            manifold_background=manifold,
            analysis_decode_mode=analysis_decode_mode,
        )
        loss = loss_b + loss_o
        loss_value = float(loss.detach().cpu())
        if initial is None:
            initial = {
                "loss_total": loss_value,
                "loss_b": float(loss_b.detach().cpu()),
                "loss_o": float(loss_o.detach().cpu()),
            }
            early_stopping_min_delta = float(tolerance) * max(abs(loss_value), 1.0)
        if not math.isfinite(loss_value):
            stop_reason = "nonfinite_loss"
            break
        if best is None or loss_value < best["loss_total"]:
            best = {
                "loss_total": loss_value,
                "loss_b": float(loss_b.detach().cpu()),
                "loss_o": float(loss_o.detach().cpu()),
                "step": int(step),
                "z": z.detach().clone(),
                "decoded": decoded.detach().clone(),
                "point_obs_count": int(diagnostics["count"]),
            }
        if (
            patience_best_loss is None
            or loss_value < patience_best_loss - early_stopping_min_delta
        ):
            patience_best_loss = loss_value
            no_improvement_steps = 0
        elif step > 0:
            no_improvement_steps += 1
        if int(patience) > 0 and no_improvement_steps >= int(patience):
            stop_reason = "best_not_improved"
            iterations = step
            break
        if step == int(max_steps):
            iterations = step
            break
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            optimizer_parameters, float(grad_clip)
        )
        last_grad_norm = float(grad_norm.detach().cpu())
        if not math.isfinite(last_grad_norm):
            stop_reason = "nonfinite_gradient"
            break
        optimizer.step()
        iterations = step + 1

    if best is None:
        raise RuntimeError("Latent optimizer did not produce any finite iterate")
    logs = {
        **initial,
        "best_loss_total": best["loss_total"],
        "best_loss_b": best["loss_b"],
        "best_loss_o": best["loss_o"],
        "best_step": best["step"],
        "iterations": iterations,
        "stop_reason": stop_reason,
        "last_grad_norm": last_grad_norm,
        "point_obs_count": best["point_obs_count"],
        "analysis_decode_mode": analysis_decode_mode,
        "ae_background_delta_rms_normalized": ae_background_delta_rms,
        "zero_increment_anchor_max_abs": (
            0.0
            if analysis_decode_mode == "background_anchored_increment"
            else float((manifold - background).abs().max().detach().cpu())
        ),
        "analysis_increment_rms_normalized": float(
            torch.sqrt((best["decoded"] - background).square().mean())
            .detach()
            .cpu()
        ),
        "early_stopping_rule": "best_loss_not_improved",
        "early_stopping_patience": int(patience),
        "early_stopping_min_delta": early_stopping_min_delta,
        "no_improvement_steps": no_improvement_steps,
        "active_branches": (
            "full_tensor"
            if requested_branches is None
            else "+".join(requested_branches)
        ),
        "latent_full_numel": int(z_b.numel()),
        "latent_optimized_numel": int(
            sum(parameter.numel() for parameter in optimizer_parameters)
        ),
        "latent_optimized_fraction": float(
            sum(parameter.numel() for parameter in optimizer_parameters)
            / max(z_b.numel(), 1)
        ),
        "bz_cholesky_numel": int(
            sum(
                item["cholesky"].numel()
                for item in bz.get("items", {}).values()
                if isinstance(item, dict) and "cholesky" in item
            )
            if isinstance(bz, dict)
            else 0
        ),
        "optimizer_wall_seconds": float(time.perf_counter() - started_at),
        "optimizer_peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(background.device))
            if background.is_cuda
            else 0
        ),
    }
    if slices is not None:
        logs.update(branch_increment_logs(best["z"], z_b.detach(), slices))
    return best["decoded"], manifold, best["z"], logs


def evaluate_traditional_cost(
    v,
    background,
    bx_std,
    kernel,
    variable_cholesky,
    observations,
    variables,
    stats,
):
    increment = apply_bx_sqrt(v, bx_std, kernel, variable_cholesky)
    analysis = background + increment
    loss_b = 0.5 * v.square().sum()
    loss_o, diagnostics = point_observation_loss(
        analysis,
        observations,
        variables,
        norm_stats=stats,
        reduction="sum",
    )
    return loss_b, loss_o, analysis, increment, diagnostics


def optimize_traditional(
    background,
    observations,
    variables,
    stats,
    bx,
    kernel,
    lr,
    max_steps,
    grad_clip,
    patience,
    tolerance,
):
    v = torch.zeros_like(background, requires_grad=True)
    optimizer = torch.optim.Adam([v], lr=float(lr))
    best = None
    patience_best_loss = None
    no_improvement_steps = 0
    early_stopping_min_delta = None
    stop_reason = "max_steps"
    last_grad_norm = None
    iterations = 0
    initial = None
    for step in range(int(max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_b, loss_o, analysis, increment, diagnostics = evaluate_traditional_cost(
            v,
            background,
            bx["std"],
            kernel,
            bx["variable_corr_cholesky"],
            observations,
            variables,
            stats,
        )
        loss = loss_b + loss_o
        loss_value = float(loss.detach().cpu())
        if initial is None:
            initial = {
                "loss_total": loss_value,
                "loss_b": float(loss_b.detach().cpu()),
                "loss_o": float(loss_o.detach().cpu()),
            }
            early_stopping_min_delta = float(tolerance) * max(abs(loss_value), 1.0)
        if not math.isfinite(loss_value):
            stop_reason = "nonfinite_loss"
            break
        if best is None or loss_value < best["loss_total"]:
            best = {
                "loss_total": loss_value,
                "loss_b": float(loss_b.detach().cpu()),
                "loss_o": float(loss_o.detach().cpu()),
                "step": int(step),
                "analysis": analysis.detach().clone(),
                "increment": increment.detach().clone(),
                "point_obs_count": int(diagnostics["count"]),
            }
        if (
            patience_best_loss is None
            or loss_value < patience_best_loss - early_stopping_min_delta
        ):
            patience_best_loss = loss_value
            no_improvement_steps = 0
        elif step > 0:
            no_improvement_steps += 1
        if int(patience) > 0 and no_improvement_steps >= int(patience):
            stop_reason = "best_not_improved"
            iterations = step
            break
        if step == int(max_steps):
            iterations = step
            break
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([v], float(grad_clip))
        last_grad_norm = float(grad_norm.detach().cpu())
        if not math.isfinite(last_grad_norm):
            stop_reason = "nonfinite_gradient"
            break
        optimizer.step()
        iterations = step + 1
    if best is None:
        raise RuntimeError("Traditional optimizer did not produce any finite iterate")
    logs = {
        **initial,
        "best_loss_total": best["loss_total"],
        "best_loss_b": best["loss_b"],
        "best_loss_o": best["loss_o"],
        "best_step": best["step"],
        "iterations": iterations,
        "stop_reason": stop_reason,
        "last_grad_norm": last_grad_norm,
        "point_obs_count": best["point_obs_count"],
        "early_stopping_rule": "best_loss_not_improved",
        "early_stopping_patience": int(patience),
        "early_stopping_min_delta": early_stopping_min_delta,
        "no_improvement_steps": no_improvement_steps,
        "increment_std": float(best["increment"].std().cpu()),
        "increment_abs_mean": float(best["increment"].abs().mean().cpu()),
    }
    return best["analysis"], logs


def serialize_records(records):
    return records


def save_outputs(records, output_dir, variables, stats, lat, lon, config, manifest, methods):
    output_dir = Path(output_dir)
    metric_rows = []
    for method in methods:
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        method_records = records[method]
        times = [np.datetime64(row["time"]) for row in method_records]
        fields = {}
        for field in ["background", "analysis", "truth", "manifold_background"]:
            values = []
            for row in method_records:
                tensor = row.get(field)
                if tensor is None:
                    values.append(np.full((len(variables), len(lat), len(lon)), np.nan, dtype=np.float32))
                else:
                    physical = destandardize_tensor(tensor, variables, stats)[0].cpu().numpy().astype(np.float32)
                    values.append(physical)
            fields[field] = np.stack(values, axis=0)
        dataset = xr.Dataset(
            {name: (("time", "variable", "lat", "lon"), values) for name, values in fields.items()},
            coords={
                "time": np.asarray(times, dtype="datetime64[ns]"),
                "variable": variables,
                "lat": np.asarray(lat, dtype=np.float32),
                "lon": np.asarray(lon, dtype=np.float32),
            },
            attrs={
                "method": method,
                "cycle_definition": "own previous two analyses -> AI 6h forecast -> optional 3DVar",
                "truth_used_as_forecast_input_after_initialization": "false",
            },
        )
        encoding = {name: {"zlib": True, "complevel": 2} for name in fields}
        dataset.to_netcdf(method_dir / "states.nc", encoding=encoding)
        dataset.close()
        log_rows = []
        for row in method_records:
            base = {"method": method, "time": row["time"], **row["optimizer_logs"]}
            log_rows.append(base)
            for variable in variables:
                bg = row["background_metrics"][variable]
                an = row["analysis_metrics"][variable]
                ob = row["observation_background"][variable]
                oa = row["observation_analysis"][variable]
                metric_rows.append(
                    {
                        "method": method,
                        "time": row["time"],
                        "variable": variable,
                        "background_rmse": bg["rmse"],
                        "analysis_rmse": an["rmse"],
                        "background_mae": bg["mae"],
                        "analysis_mae": an["mae"],
                        "background_bias": bg["bias"],
                        "analysis_bias": an["bias"],
                        "background_corr": bg["corr"],
                        "analysis_corr": an["corr"],
                        "rmse_reduction_percent": 100.0 * (bg["rmse"] - an["rmse"]) / max(bg["rmse"], 1e-12),
                        "obs_count": ob["count"],
                        "ob_rmse": ob["rmse"],
                        "oa_rmse": oa["rmse"],
                        "ob_chi2": ob["chi2"],
                        "oa_chi2": oa["chi2"],
                    }
                )
        pd.DataFrame(log_rows).to_csv(method_dir / "optimizer_by_cycle.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics_by_cycle_variable.csv", index=False)
    summary = (
        metrics.groupby(["method", "variable"], as_index=False)
        .agg(
            background_rmse=("background_rmse", "mean"),
            analysis_rmse=("analysis_rmse", "mean"),
            background_corr=("background_corr", "mean"),
            analysis_corr=("analysis_corr", "mean"),
            rmse_reduction_percent=("rmse_reduction_percent", "mean"),
            ob_chi2=("ob_chi2", "mean"),
            oa_chi2=("oa_chi2", "mean"),
        )
    )
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    atomic_json(manifest, output_dir / "manifest.json")


def main():
    args = parse_args()
    config = load_yaml(args.config)
    methods = list(config.get("methods", METHODS))
    if not methods or len(methods) != len(set(methods)):
        raise ValueError(f"methods must be a non-empty list without duplicates: {methods}")
    unknown_methods = sorted(set(methods) - set(METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown methods {unknown_methods}; available={METHODS}")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and args.resume_from is None and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(config.get("cpu_threads", 16)))
    device = resolve_device(args.device)
    deterministic_algorithms = bool(config.get("deterministic_algorithms", False))
    set_seed(int(config.get("seed", 42)), deterministic=deterministic_algorithms)

    variables = list(config["variables"])
    forecast_cfg = load_forecast_config(config["forecast_config"])
    use_lda = "LDA" in methods
    use_physp = "PhySP_DA" in methods
    use_traditional = "Traditional" in methods
    lda_cfg = load_ae_config(config["lda_config"]) if use_lda else None
    phy_cfg = load_ae_config(config["physp_config"]) if use_physp else None
    if variables != list(forecast_cfg["target_vars"]):
        raise ValueError("Variable order differs from forecast model")
    if lda_cfg is not None and variables != list(lda_cfg["target_vars"]):
        raise ValueError("Variable order differs from LDA model")
    if phy_cfg is not None and variables != list(phy_cfg["target_vars"]):
        raise ValueError("Variable order differs from OCS-LDA model")
    with open(config["norm_stats"], "r", encoding="utf-8") as stream:
        norm_stats_raw = json.load(stream)
    stats = norm_stats_raw.get("variables", norm_stats_raw)

    print(f"Loading models on {device}", flush=True)
    forecast = load_forecast_model(forecast_cfg, config["forecast_checkpoint"], device)
    lda = load_ae_model(lda_cfg, config["lda_checkpoint"], device) if use_lda else None
    physp = load_ae_model(phy_cfg, config["physp_checkpoint"], device) if use_physp else None
    forecast.eval()
    if lda is not None:
        lda.eval()
    if physp is not None:
        physp.eval()
    traditional_variance_mode = config.get("traditional_variance_mode", "channel")
    lda_bz_mode = config.get("lda_bz_mode", "channel_covariance")
    physp_bz_mode = config.get("physp_bz_mode", "channel_covariance")
    physp_active_branches = normalize_active_branches(
        config.get("physp_active_branches")
    )
    physp_analysis_decode_mode = str(
        config.get("physp_analysis_decode_mode", "absolute_latent_state")
    )
    if physp_analysis_decode_mode not in {
        "absolute_latent_state",
        "background_anchored_increment",
    }:
        raise ValueError(
            "physp_analysis_decode_mode must be 'absolute_latent_state' or "
            "'background_anchored_increment'"
        )
    physp_branch_slices = branch_slices_from_model(physp) if use_physp else None
    if use_physp and physp_bz_mode == "semantic_branch_block":
        if physp_branch_slices is None:
            raise ValueError(
                "semantic_branch_block requires a branch-structured PhySP_DA model."
            )
        if physp_active_branches is None:
            physp_active_branches = list(physp_branch_slices)
        unknown = sorted(set(physp_active_branches) - set(physp_branch_slices))
        if unknown:
            raise ValueError(
                f"Unknown physp_active_branches {unknown}; "
                f"available={list(physp_branch_slices)}"
            )
    bx = load_physical_bx(
        config["traditional_bx"],
        variables,
        device,
        scale=float(config["bx_scale"]),
        variance_mode=traditional_variance_mode,
    ) if use_traditional else None
    lda_bz = select_latent_bz_mode(
        load_latent_bz(config["lda_bz"], device, scale=float(config["lda_bz_scale"])),
        lda_bz_mode,
        "LDA",
    ) if use_lda else None
    phy_bz_loaded = load_latent_bz(
        config["physp_bz"],
        device,
        scale=float(config["physp_bz_scale"]),
        item_scales=config.get("physp_bz_branch_scales"),
        item_names=(
            physp_active_branches
            if physp_bz_mode == "semantic_branch_block"
            else None
        ),
        load_covariance=physp_bz_mode != "semantic_branch_block",
    ) if use_physp else None
    phy_bz = select_latent_bz_mode(
        phy_bz_loaded,
        physp_bz_mode,
        "PhySP_DA",
        branch_names=physp_active_branches,
    ) if use_physp else None
    kernel = gaussian_kernel2d(
        config["gaussian_sigma"], config["gaussian_radius"], device, torch.float32
    ) if use_traditional else None
    # B^(1/2) must preserve the fitted channel standard deviation.  L2
    # normalization gives unit variance for smoothed white-noise control.
    if kernel is not None:
        kernel = kernel / torch.sqrt(kernel.square().sum()).clamp_min(1e-12)

    init_ds = xr.open_dataset(config["initialization_file"])
    truth_ds = xr.open_dataset(config["nature_file"])
    lat = truth_ds.lat.values.astype(np.float32)
    lon = truth_ds.lon.values.astype(np.float32)
    initialization_times = [
        pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%S")
        for value in config["initialization_times"]
    ]
    init_states = []
    for value in initialization_times:
        physical = read_physical_state(init_ds, value, variables)
        init_states.append(torch.from_numpy(standardize(physical, variables, stats)).unsqueeze(0).to(device))
    init_ds.close()
    obs_loader = GdasOssePointObsLoader(
        config["observation_file"],
        variables,
        lat,
        lon,
        network=config.get("observation_network", "extended"),
        r_scale=float(config.get("r_scale", 1.0)),
        variables=config.get("observation_variables"),
    )

    cycle_times = pd.date_range(config["start_time"], config["end_time"], freq="6h")
    if len(cycle_times) != int(config.get("expected_cycles", len(cycle_times))):
        raise ValueError("Unexpected cycle count")
    if args.resume_from:
        state = torch.load(args.resume_from, map_location=device, weights_only=False)
        histories = state["histories"]
        histories = {name: [item.to(device) for item in pair] for name, pair in histories.items()}
        records = state["records"]
        start_index = int(state["completed_cycles"])
        print(f"Resuming after {start_index} cycles from {args.resume_from}", flush=True)
    else:
        histories = {name: [init_states[0].clone(), init_states[1].clone()] for name in methods}
        records = {name: [] for name in methods}
        start_index = 0

    stop_after = len(cycle_times) if args.stop_after <= 0 else min(args.stop_after, len(cycle_times))
    checkpoint_mode = str(config.get("checkpoint_mode", "per_cycle")).lower()
    checkpoint_interval = int(config.get("checkpoint_interval_cycles", 1))
    # Validate before the expensive cycling loop begins.
    checkpoint_save_kind(1, stop_after, len(cycle_times), checkpoint_mode, checkpoint_interval)
    last_checkpoint_path = Path(args.resume_from) if args.resume_from else None
    manifest = {
        "config": args.config,
        "device": str(device),
        "methods": methods,
        "deterministic_algorithms": deterministic_algorithms,
        "variables": variables,
        "observation_variables": list(config.get("observation_variables", variables)),
        "observation_network": str(config.get("observation_network", "extended")),
        "r_scale": float(config.get("r_scale", 1.0)),
        "initialization_times": initialization_times,
        "cycle_times": [value.strftime("%Y-%m-%dT%H:%M:%S") for value in cycle_times],
        "forecast_checkpoint_sha256": file_sha256(config["forecast_checkpoint"]),
        "lda_checkpoint_sha256": file_sha256(config["lda_checkpoint"]) if use_lda else None,
        "physp_checkpoint_sha256": file_sha256(config["physp_checkpoint"]) if use_physp else None,
        "physp_model_arch": str(phy_cfg.get("model_arch", "unknown")) if use_physp else None,
        "physp_hub_gates": collect_physp_hub_gates(physp) if use_physp else None,
        "traditional_bx_sha256": file_sha256(config["traditional_bx"]) if use_traditional else None,
        "lda_bz_sha256": file_sha256(config["lda_bz"]) if use_lda else None,
        "physp_bz_sha256": file_sha256(config["physp_bz"]) if use_physp else None,
        "truth_after_initialization_used_as_forecast_input": False,
        "point_observation_reduction": "sum",
        "background_loss_reduction": "sum",
        "early_stopping_rule": "best loss not improved by tolerance-scaled min_delta for patience consecutive evaluations",
        "early_stopping_patience": int(config["early_stopping_patience"]),
        "early_stopping_tolerance": float(config["early_stopping_tolerance"]),
        "checkpoint_mode": checkpoint_mode,
        "checkpoint_interval_cycles": checkpoint_interval,
        "checkpoint_retention": (
            "overwrite latest.pt every interval; retain only numbered final checkpoint"
            if checkpoint_mode == "rolling"
            else "retain one numbered checkpoint per cycle"
        ),
        "gaussian_kernel_normalization": "L2 for B^(1/2) variance preservation",
        "traditional_variance_mode": bx["variance_mode"] if use_traditional else None,
        "traditional_std_shape": list(bx["std"].shape) if use_traditional else None,
        "traditional_std_minmax": [float(bx["std"].min().cpu()), float(bx["std"].max().cpu())] if use_traditional else None,
        "lda_bz_mode": str(lda_bz_mode),
        "lda_bz_shape": list(lda_bz.shape) if torch.is_tensor(lda_bz) else None,
        "physp_bz_mode": str(physp_bz_mode),
        "physp_bz_shape": list(phy_bz.shape) if torch.is_tensor(phy_bz) else None,
        "physp_bz_semantic_branch_names": phy_bz.get("semantic_branch_names") if isinstance(phy_bz, dict) else None,
        "physp_bz_semantic_branch_sizes": phy_bz.get("semantic_branch_sizes") if isinstance(phy_bz, dict) else None,
        "physp_active_branches": physp_active_branches,
        "physp_analysis_decode_mode": physp_analysis_decode_mode,
        "physp_bz_loaded_item_names": (
            phy_bz.get("loaded_item_names")
            if isinstance(phy_bz, dict)
            else None
        ),
        "physp_bz_covariance_loaded": (
            phy_bz.get("covariance_loaded")
            if isinstance(phy_bz, dict)
            else None
        ),
        "physp_bz_branch_scale_multipliers": (
            phy_bz.get("latent_bz_item_scale_multipliers")
            if isinstance(phy_bz, dict)
            else None
        ),
        "physp_bz_effective_branch_scales": (
            phy_bz.get("latent_bz_effective_item_scales")
            if isinstance(phy_bz, dict)
            else None
        ),
    }

    for cycle_index in range(start_index, stop_after):
        cycle_time = cycle_times[cycle_index]
        time_string = cycle_time.strftime("%Y-%m-%dT%H:%M:%S")
        print(f"Cycle {cycle_index + 1}/{len(cycle_times)} {time_string}", flush=True)
        physical_truth = read_physical_state(truth_ds, time_string, variables)
        truth = torch.from_numpy(standardize(physical_truth, variables, stats)).unsqueeze(0).to(device)
        point_obs = obs_loader.load_batch([time_string], device)
        cycle_backgrounds = {}
        for method in methods:
            previous, current = histories[method]
            cycle_backgrounds[method] = forecast_one_step(
                forecast,
                previous,
                current,
                time_string,
                lat,
                lon,
                forecast_cfg.get("forecast_time_feature_mode", "local_solar"),
            )
        if cycle_index == 0:
            reference = cycle_backgrounds[methods[0]]
            for method in methods[1:]:
                max_delta = float((cycle_backgrounds[method] - reference).abs().max().cpu())
                if max_delta > 1e-7:
                    raise RuntimeError(f"First-cycle backgrounds differ for {method}: {max_delta}")

        analyses = {}
        manifolds = {name: None for name in methods}
        logs = {}
        if "AI_FreeRun" in methods:
            analyses["AI_FreeRun"] = cycle_backgrounds["AI_FreeRun"]
            logs["AI_FreeRun"] = {
                "iterations": 0,
                "stop_reason": "no_assimilation",
                "initial_loss_total": None,
                "best_loss_total": None,
            }
        if "Traditional" in methods:
            analyses["Traditional"], logs["Traditional"] = optimize_traditional(
                cycle_backgrounds["Traditional"],
                point_obs,
                variables,
                stats,
                bx,
                kernel,
                config.get("traditional_optimizer_lr", config["optimizer_lr"]),
                config["optimizer_max_steps"],
                config["optimizer_grad_clip"],
                config["early_stopping_patience"],
                config["early_stopping_tolerance"],
            )
        if "LDA" in methods:
            analyses["LDA"], manifolds["LDA"], _, logs["LDA"] = optimize_latent(
                lda,
                cycle_backgrounds["LDA"],
                point_obs,
                variables,
                stats,
                lda_bz,
                config.get("lda_optimizer_lr", config["optimizer_lr"]),
                config["optimizer_max_steps"],
                config["optimizer_grad_clip"],
                config["early_stopping_patience"],
                config["early_stopping_tolerance"],
            )
        if "PhySP_DA" in methods:
            analyses["PhySP_DA"], manifolds["PhySP_DA"], _, logs["PhySP_DA"] = optimize_latent(
                physp,
                cycle_backgrounds["PhySP_DA"],
                point_obs,
                variables,
                stats,
                phy_bz,
                config.get("physp_optimizer_lr", config["optimizer_lr"]),
                config["optimizer_max_steps"],
                config["optimizer_grad_clip"],
                config["early_stopping_patience"],
                config["early_stopping_tolerance"],
                active_branches=physp_active_branches,
                analysis_decode_mode=physp_analysis_decode_mode,
            )

        for method in methods:
            background = cycle_backgrounds[method]
            analysis = analyses[method].detach()
            if not bool(torch.isfinite(background).all() and torch.isfinite(analysis).all()):
                raise RuntimeError(f"Non-finite state for {method} at {time_string}")
            records[method].append(
                {
                    "time": time_string,
                    "background": background.detach().cpu(),
                    "analysis": analysis.detach().cpu(),
                    "truth": truth.detach().cpu(),
                    "manifold_background": None if manifolds[method] is None else manifolds[method].detach().cpu(),
                    "background_metrics": field_metrics(background, truth, variables, stats),
                    "analysis_metrics": field_metrics(analysis, truth, variables, stats),
                    "observation_background": observation_metrics(background, point_obs[0], variables, stats),
                    "observation_analysis": observation_metrics(analysis, point_obs[0], variables, stats),
                    "optimizer_logs": logs[method],
                    "forecast_input_times": [
                        initialization_times[cycle_index] if cycle_index < 2 else records[method][cycle_index - 2]["time"],
                        initialization_times[cycle_index + 1] if cycle_index == 0 else records[method][cycle_index - 1]["time"],
                    ] if cycle_index == 0 else [
                        initialization_times[1] if cycle_index == 1 else records[method][cycle_index - 2]["time"],
                        records[method][cycle_index - 1]["time"],
                    ],
                }
            )
            histories[method] = [histories[method][1].detach(), analysis]

        completed_cycles = cycle_index + 1
        save_kind = checkpoint_save_kind(
            completed_cycles,
            stop_after,
            len(cycle_times),
            checkpoint_mode,
            checkpoint_interval,
        )
        if save_kind is not None:
            checkpoint = {
                "completed_cycles": completed_cycles,
                "histories": {
                    name: [item.detach().cpu() for item in pair]
                    for name, pair in histories.items()
                },
                "records": serialize_records(records),
                "manifest": manifest,
            }
            if save_kind == "latest":
                checkpoint_path = output_dir / "checkpoints" / "latest.pt"
            else:
                checkpoint_path = (
                    output_dir / "checkpoints" / f"cycle_{completed_cycles:03d}.pt"
                )
            atomic_torch_save(checkpoint, checkpoint_path)
            last_checkpoint_path = checkpoint_path
            print(f"Saved {checkpoint_path}", flush=True)
            if save_kind == "final" and checkpoint_mode == "rolling":
                obsolete_latest = output_dir / "checkpoints" / "latest.pt"
                if obsolete_latest.exists():
                    obsolete_latest.unlink()
                    print(f"Removed obsolete {obsolete_latest}", flush=True)

    truth_ds.close()
    obs_loader.close()
    save_outputs(records, output_dir, variables, stats, lat, lon, config, manifest, methods)
    atomic_json(
        {
            "status": "complete" if stop_after == len(cycle_times) else "partial",
            "completed_cycles": stop_after,
            "expected_cycles": len(cycle_times),
            "last_checkpoint": (
                str(last_checkpoint_path) if last_checkpoint_path is not None else None
            ),
            "checkpoint_mode": checkpoint_mode,
            "checkpoint_interval_cycles": checkpoint_interval,
        },
        output_dir / "RUN_STATUS.json",
    )
    print(f"Finished {stop_after}/{len(cycle_times)} cycles", flush=True)


if __name__ == "__main__":
    main()
