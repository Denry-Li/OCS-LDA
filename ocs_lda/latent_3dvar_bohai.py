import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
import yaml
from tqdm import tqdm

from ocs_lda.dataset_bohai import build_dataloader
from ocs_lda.train_bohai import AttrDict, channelwise_metrics, configure_loss_from_type
from utils.point_observation_operator import (
    GdasOssePointObsLoader,
    point_observation_loss,
)


CCMP_TO_TARGET = {
    "ccmp_u10": "era5_u10",
    "ccmp_v10": "era5_v10",
}

RAW_GFS_ORDER = ["gfs_t2m", "gfs_u10", "gfs_v10", "gfs_d2m", "gfs_sp"]
TARGET_TO_GFS = {
    "era5_t2m": "gfs_t2m",
    "era5_u10": "gfs_u10",
    "era5_v10": "gfs_v10",
    "era5_d2m": "gfs_d2m",
    "era5_sp": "gfs_sp",
}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return configure_loss_from_type(AttrDict(yaml.safe_load(f)))


def resolve_device(device_arg):
    device_arg = str(device_arg).lower()
    if device_arg == "auto":
        device_arg = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda":
        device_arg = "cuda:0"
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(int(device_arg.split(":")[1]))
    return torch.device(device_arg)


def strip_module_prefix(state):
    if not any(k.startswith("module.") for k in state):
        return state
    return {k.replace("module.", "", 1): v for k, v in state.items()}


def load_model(cfg, checkpoint_path, device):
    model_arch = str(cfg.get("model_arch", "swim_ir_ae")).lower()
    if model_arch == "swim_ir_ae":
        from models.SwimIR_AE import SwimIR_AE

        model = SwimIR_AE(cfg).to(device)
    elif model_arch in {"shared_private_swim_ir_ae", "shared_private_swimir_ae"}:
        from models.SharedPrivateSwimIR_AE import SharedPrivateSwimIR_AE

        model = SharedPrivateSwimIR_AE(cfg).to(device)
    elif model_arch == "lda_vit_ae":
        from models.LDA_ViT_AE import LDA_ViT_AE

        model = LDA_ViT_AE(cfg).to(device)
    elif model_arch == "lda141_ae":
        from models.LDA141_AE import LDA141_AE

        model = LDA141_AE(cfg).to(device)
    elif model_arch in {"structured_swim_ae", "structed_swim_ae"}:
        from models.StructedSwimAE import StructuredSwinAE

        model = StructuredSwinAE(cfg).to(device)
    elif model_arch in {
        "structured_swim_ae_refine",
        "structed_swim_ae_refine",
    }:
        from models.ocs_lda_blocks import StructuredSwinAERefine

        model = StructuredSwinAERefine(cfg).to(device)
    elif model_arch in {"structured_swim_ae_v2", "structured_swin_ae_v2"}:
        from models.StructuredSwinAEv2 import StructuredSwinAEv2

        model = StructuredSwinAEv2(cfg).to(device)
    elif model_arch == "physp_aev2":
        from models.PhySP_AEv2 import PhySP_AEv2

        model = PhySP_AEv2(cfg).to(device)
    elif model_arch == "physp_aev3":
        from models.ocs_lda import PhySP_AEv3

        model = PhySP_AEv3(cfg).to(device)
    elif model_arch == "physp_aev4":
        from models.ocs_lda import PhySP_AEv4

        model = PhySP_AEv4(cfg).to(device)
    else:
        raise ValueError(f"Unsupported AE model_arch={model_arch!r}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = strip_module_prefix(ckpt["model_state"])
    model.load_state_dict(state)
    if hasattr(model, "freeze_ae"):
        return model.freeze_ae()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def encode_det_with_size(model, x):
    encoded = model.encode_det(x)
    if isinstance(encoded, tuple):
        return encoded
    return encoded, x.shape[-2:]


def decode_det_with_size(model, z, output_size):
    try:
        return model.decode_det(z, output_size)
    except TypeError:
        return model.decode_det(z)


class CcmpGridObsLoader:
    def __init__(self, root, norm_stats, target_vars, use_conf_weight=False):
        self.root = Path(root)
        self.norm_stats = norm_stats
        self.target_vars = list(target_vars)
        self.use_conf_weight = bool(use_conf_weight)
        self.datasets = {}
        self.time_indices = {}

    def _open_year(self, year):
        year = int(year)
        if year not in self.datasets:
            path = self.root / f"ccmp_grid_{year}_standard.nc"
            if not path.exists():
                raise FileNotFoundError(f"Missing CCMP grid file: {path}")
            ds = xr.open_dataset(path)
            self.datasets[year] = ds
            self.time_indices[year] = {
                np.datetime_as_string(value, unit="s"): i
                for i, value in enumerate(ds["time"].values)
            }
        return self.datasets[year], self.time_indices[year]

    def close(self):
        for ds in self.datasets.values():
            ds.close()
        self.datasets.clear()
        self.time_indices.clear()

    def _standardize_like_target(self, arr, target_var):
        stats = self.norm_stats[target_var]
        mean = float(stats["mean"])
        std = float(stats["std"])
        return ((arr - mean) / (std + 1e-6)).astype(np.float32)

    def load_batch(self, times, device):
        obs_arrays = []
        weight_arrays = []
        for time_value in times:
            timestamp = pd.Timestamp(str(time_value))
            time_key = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
            ds, time_index = self._open_year(timestamp.year)
            if time_key not in time_index:
                raise KeyError(f"{time_key} not found in CCMP grid year {timestamp.year}")
            ti = time_index[time_key]

            values = []
            weights = []
            for ccmp_var, target_var in CCMP_TO_TARGET.items():
                arr = ds[ccmp_var].isel(time=ti).values.astype(np.float32)
                obs = self._standardize_like_target(arr, target_var)

                if "ccmp_mask" in ds:
                    mask = ds["ccmp_mask"].isel(time=ti).values.astype(np.float32)
                else:
                    mask = np.isfinite(arr).astype(np.float32)
                mask = (mask > 0).astype(np.float32)

                if self.use_conf_weight and "ccmp_conf" in ds:
                    conf = ds["ccmp_conf"].isel(time=ti).values.astype(np.float32)
                    weight = mask * np.nan_to_num(conf, nan=0.0).clip(min=0.0)
                else:
                    weight = mask

                values.append(np.nan_to_num(obs, nan=0.0).astype(np.float32))
                weights.append(np.nan_to_num(weight, nan=0.0).astype(np.float32))

            obs_arrays.append(np.stack(values, axis=0))
            weight_arrays.append(np.stack(weights, axis=0))

        obs = torch.from_numpy(np.stack(obs_arrays, axis=0)).to(device)
        weight = torch.from_numpy(np.stack(weight_arrays, axis=0)).to(device)
        return obs, weight


class IcoadsPointObsLoader:
    def __init__(
        self,
        root,
        norm_stats,
        target_var="era5_t2m",
        window_hours=3.0,
        qc="strict",
    ):
        self.root = Path(root)
        self.norm_stats = norm_stats
        self.target_var = target_var
        self.window = pd.Timedelta(hours=float(window_hours))
        self.qc = str(qc).lower()
        self.tables = {}
        if self.target_var not in self.norm_stats:
            raise KeyError(f"{self.target_var!r} not found in normalization statistics.")
        if self.qc not in {"strict", "phys", "none"}:
            raise ValueError(f"Unsupported ICOADS QC mode {qc!r}; use strict, phys, or none.")

    def _load_year(self, year):
        year = int(year)
        if year not in self.tables:
            path = self.root / f"icoads_region_{year}.parquet"
            if not path.exists():
                self.tables[year] = pd.DataFrame()
            else:
                columns = [
                    "time",
                    "lat",
                    "lon",
                    "AT",
                    "AT_valid_phys",
                    "AT_qc_good",
                    "AT_trim_good",
                ]
                df = pd.read_parquet(path, columns=columns)
                df["time"] = pd.to_datetime(df["time"])
                self.tables[year] = df
        return self.tables[year]

    def close(self):
        self.tables.clear()

    def _filter_qc(self, df):
        if df.empty:
            return df
        mask = np.isfinite(df["AT"].to_numpy(dtype=np.float32))
        if self.qc in {"strict", "phys"}:
            mask = mask & df["AT_valid_phys"].fillna(False).to_numpy(dtype=bool)
        if self.qc == "strict":
            mask = (
                mask
                & df["AT_qc_good"].fillna(False).to_numpy(dtype=bool)
                & df["AT_trim_good"].fillna(False).to_numpy(dtype=bool)
            )
        return df.loc[mask]

    def _standardize_t2m(self, values):
        stats = self.norm_stats[self.target_var]
        mean = float(stats["mean"])
        std = float(stats["std"])
        return ((values - mean) / (std + 1e-6)).astype(np.float32)

    def load_batch(self, times, device):
        batches = []
        for time_value in times:
            timestamp = pd.Timestamp(str(time_value))
            start = timestamp - self.window
            end = timestamp + self.window
            parts = []
            for year in range(start.year, end.year + 1):
                df = self._load_year(year)
                if df.empty:
                    continue
                sub = df[(df["time"] >= start) & (df["time"] <= end)]
                if not sub.empty:
                    parts.append(sub)
            if parts:
                obs_df = self._filter_qc(pd.concat(parts, ignore_index=True))
            else:
                obs_df = pd.DataFrame()

            if obs_df.empty:
                batches.append(
                    {
                        "lat": torch.empty(0, device=device, dtype=torch.float32),
                        "lon": torch.empty(0, device=device, dtype=torch.float32),
                        "value": torch.empty(0, device=device, dtype=torch.float32),
                        "dt_hours": torch.empty(0, device=device, dtype=torch.float32),
                    }
                )
                continue

            values = self._standardize_t2m(obs_df["AT"].to_numpy(dtype=np.float32))
            dt_hours = (
                (obs_df["time"] - timestamp)
                .dt.total_seconds()
                .abs()
                .to_numpy(dtype=np.float32)
                / 3600.0
            )
            batches.append(
                {
                    "lat": torch.from_numpy(obs_df["lat"].to_numpy(dtype=np.float32)).to(device),
                    "lon": torch.from_numpy(obs_df["lon"].to_numpy(dtype=np.float32)).to(device),
                    "value": torch.from_numpy(values).to(device),
                    "dt_hours": torch.from_numpy(dt_hours).to(device),
                }
            )
        return batches


def corr_tensor(pred, target, eps=1e-8):
    x = pred.reshape(-1)
    y = target.reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).sum() / (torch.sqrt((x * x).sum() * (y * y).sum()) + eps)


def metric_dict(pred, target, prefix, channel_names):
    out = {
        f"{prefix}_rmse_mean": torch.sqrt(F.mse_loss(pred, target)).item(),
        f"{prefix}_mae_mean": F.l1_loss(pred, target).item(),
        f"{prefix}_corr_mean": corr_tensor(pred, target).item(),
    }
    metrics = channelwise_metrics(pred, target, channel_names)
    for key, value in metrics.items():
        out[f"{prefix}_{key}"] = float(value.item())
    return out


def standardize_raw_background_as_target(raw_background, norm_stats, target_vars):
    if raw_background.shape[1] != len(RAW_GFS_ORDER):
        raise ValueError(
            "raw_background channel count must match RAW_GFS_ORDER length: "
            f"{raw_background.shape[1]} != {len(RAW_GFS_ORDER)}"
        )

    raw_lookup = {name: ci for ci, name in enumerate(RAW_GFS_ORDER)}
    standardized = []
    for target_var in target_vars:
        if target_var not in TARGET_TO_GFS:
            raise KeyError(
                f"Cannot map target variable {target_var!r} to a raw GFS channel. "
                f"Known target variables: {sorted(TARGET_TO_GFS)}"
            )
        gfs_var = TARGET_TO_GFS[target_var]
        if gfs_var not in raw_lookup:
            raise KeyError(f"Missing {gfs_var!r} in RAW_GFS_ORDER={RAW_GFS_ORDER}")
        ci = raw_lookup[gfs_var]
        stats = norm_stats[target_var]
        mean = float(stats["mean"])
        std = float(stats["std"])
        standardized.append((raw_background[:, ci] - mean) / (std + 1e-6))
    return torch.stack(standardized, dim=1).to(dtype=torch.float32)


def destandardize_dynamic_fields(fields, norm_stats, target_vars):
    """Convert target-standardized BCHW fields back to physical units."""
    physical = []
    for ci, target_var in enumerate(target_vars):
        stats = norm_stats[target_var]
        mean = float(stats["mean"])
        std = float(stats["std"])
        physical.append(fields[:, ci] * std + mean)
    return torch.stack(physical, dim=1).to(dtype=torch.float32)


def save_physical_fields_netcdf(
    path,
    field_store,
    times,
    lat,
    lon,
    target_vars,
    metadata,
):
    units = {
        "era5_u10": "m s-1",
        "era5_v10": "m s-1",
        "era5_t2m": "degC",
        "era5_d2m": "degC",
        "era5_sp": "hPa",
    }
    data_vars = {}
    for field_name, samples in field_store.items():
        values = np.concatenate(samples, axis=0).astype(np.float32, copy=False)
        for ci, target_var in enumerate(target_vars):
            short_name = target_var.removeprefix("era5_")
            variable_name = f"{field_name}_{short_name}"
            data_vars[variable_name] = (
                ("time", "lat", "lon"),
                values[:, ci],
                {
                    "units": units.get(target_var, "unknown"),
                    "source_target_variable": target_var,
                    "field_role": field_name,
                },
            )
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": np.asarray(times, dtype="datetime64[ns]"),
            "lat": np.asarray(lat, dtype=np.float32),
            "lon": np.asarray(lon, dtype=np.float32),
        },
        attrs=metadata,
    )
    encoding = {
        name: {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "_FillValue": np.float32(np.nan),
        }
        for name in dataset.data_vars
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path, encoding=encoding)


def append_static_channels_for_model(dynamic_fields, batch, cfg):
    static_vars = list(cfg.get("static_vars", []))
    if not static_vars:
        return dynamic_fields
    input_tensor = batch["input"].to(
        dynamic_fields.device,
        dtype=dynamic_fields.dtype,
        non_blocking=True,
    )
    n_dynamic = len(cfg.get("input_vars", [])) + len(cfg.get("ccmp_vars", []))
    n_static = len(static_vars)
    static = input_tensor[:, n_dynamic : n_dynamic + n_static]
    if static.shape[1] != n_static:
        raise ValueError(
            f"Expected {n_static} static channels, got {static.shape[1]}."
        )
    return torch.cat([dynamic_fields, static], dim=1)


def dynamic_channels_for_metrics(fields, cfg):
    return fields[:, : len(cfg["target_vars"])]


def load_latent_bz(
    path,
    device,
    floor=1e-6,
    scale=1.0,
    item_scales=None,
    item_names=None,
    load_covariance=True,
):
    if path is None:
        return None
    selected_names = None if item_names is None else {str(name) for name in item_names}
    normalized_item_scales = {
        str(name): float(value) for name, value in (item_scales or {}).items()
    }
    invalid_scales = {
        name: value
        for name, value in normalized_item_scales.items()
        if not math.isfinite(value) or value <= 0.0
    }
    if invalid_scales:
        raise ValueError(
            f"Latent B_z item scales must be positive and finite: {invalid_scales}"
        )
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError(f"Latent B_z global scale must be positive and finite, got {scale}")
    # Select on CPU first so inactive semantic blocks and the legacy full
    # covariance are never materialized on the accelerator.
    map_location = "cpu" if selected_names is not None else device
    data = torch.load(path, map_location=map_location)
    if isinstance(data, dict):
        if data.get("type") == "latent_channel_cov":
            loaded = dict(data)
            available_names = {str(name) for name in data["items"]}
            unknown_scales = sorted(set(normalized_item_scales) - available_names)
            if unknown_scales:
                raise KeyError(
                    f"{path} does not contain scaled latent B_z items {unknown_scales}; "
                    f"available={sorted(available_names)}"
                )
            items = {}
            effective_item_scales = {}
            for name, item in data["items"].items():
                name = str(name)
                if selected_names is not None and name not in selected_names:
                    continue
                effective_scale = float(scale) * normalized_item_scales.get(name, 1.0)
                effective_item_scales[name] = effective_scale
                item_out = dict(item)
                if load_covariance and "cov_loaded" in item:
                    item_out["cov_loaded"] = (
                        torch.as_tensor(item["cov_loaded"], device=device, dtype=torch.float32)
                        * effective_scale
                    )
                else:
                    item_out.pop("cov_loaded", None)
                if "cholesky" in item:
                    item_out["cholesky"] = (
                        torch.as_tensor(item["cholesky"], device=device, dtype=torch.float32)
                        * (effective_scale ** 0.5)
                    )
                if "diag_var_spatial" in item:
                    item_out["diag_var_spatial"] = torch.as_tensor(
                        item["diag_var_spatial"], device=device, dtype=torch.float32
                    ).clamp_min(float(floor)) * effective_scale
                items[name] = item_out
            if selected_names is not None:
                missing = sorted(selected_names - set(items))
                if missing:
                    raise KeyError(
                        f"{path} does not contain requested latent B_z items {missing}; "
                        f"available={sorted(str(name) for name in data['items'])}"
                    )
            loaded["items"] = items
            loaded["latent_bz_scale"] = float(scale)
            loaded["latent_bz_item_scale_multipliers"] = {
                name: normalized_item_scales.get(name, 1.0) for name in sorted(items)
            }
            loaded["latent_bz_effective_item_scales"] = {
                name: effective_item_scales[name] for name in sorted(items)
            }
            loaded["loaded_item_names"] = sorted(items)
            loaded["covariance_loaded"] = bool(load_covariance)
            return loaded
        if "var" in data:
            bz = data["var"]
        elif "second_moment" in data:
            bz = data["second_moment"]
        else:
            raise KeyError(f"{path} must contain 'var' or 'second_moment'")
    else:
        bz = data
    if normalized_item_scales:
        raise ValueError(
            "Per-item latent B_z scales require a latent_channel_cov payload."
        )
    bz = torch.as_tensor(bz, device=device, dtype=torch.float32)
    return bz.clamp_min(float(floor)) * float(scale)


def load_obs_r_diag(path, device, key="r_diag", floor=1e-6, ceiling=None, scale=1.0):
    if path is None:
        return None
    data = torch.load(path, map_location=device)
    if isinstance(data, dict):
        if key not in data:
            available = ", ".join(sorted(str(k) for k in data.keys()))
            raise KeyError(f"{path} does not contain {key!r}. Available keys: {available}")
        obs_r = data[key]
    else:
        obs_r = data
    obs_r = torch.as_tensor(obs_r, device=device, dtype=torch.float32) * float(scale)
    obs_r = obs_r.clamp_min(float(floor))
    if ceiling is not None and float(ceiling) > 0:
        obs_r = obs_r.clamp_max(float(ceiling))
    return obs_r


def latent_background_loss(z, z_b, background_error, reduction="mean", eps=1e-8):
    if isinstance(background_error, dict) and background_error.get("type") == "latent_channel_cov":
        items = background_error["items"]
        if "latent" not in items:
            raise KeyError("Channel B_z for full latent optimization must contain items['latent'].")
        return channel_background_loss(z, z_b, items["latent"], reduction=reduction, eps=eps)
    if torch.is_tensor(background_error):
        denom = background_error.to(device=z.device, dtype=z.dtype)
        while denom.ndim < z.ndim:
            denom = denom.unsqueeze(0)
    else:
        denom = float(background_error)
    loss = 0.5 * ((z - z_b) ** 2 / (denom + eps))
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unsupported background loss reduction: {reduction}")


def channel_background_loss(z, z_b, channel_bz_item, reduction="mean", eps=1e-8):
    dz = z - z_b
    b, c, h, w = dz.shape
    chol = channel_bz_item["cholesky"].to(device=dz.device, dtype=dz.dtype)
    if chol.shape != (c, c):
        raise ValueError(f"Channel B_z Cholesky shape {tuple(chol.shape)} does not match latent channels {c}.")
    rows = dz.permute(0, 2, 3, 1).reshape(-1, c)
    # B = L L^T, therefore dz^T B^-1 dz = ||L^-1 dz||^2.  A single
    # triangular solve is cheaper than cholesky_solve followed by a dot
    # product and is mathematically identical.
    whitened = torch.linalg.solve_triangular(
        chol,
        rows.transpose(0, 1).contiguous(),
        upper=False,
    )
    loss = 0.5 * whitened.square().sum(dim=0)
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unsupported background loss reduction: {reduction}")


def branch_slices_from_model(model):
    if not hasattr(model, "branch_latent") or not hasattr(model.branch_latent, "branch_channels"):
        return None
    slices = {}
    start = 0
    for name, channels in model.branch_latent.branch_channels.items():
        end = start + int(channels)
        slices[str(name)] = slice(start, end)
        start = end
    return slices


def normalize_active_branches(active_branches):
    if active_branches is None:
        return None
    active = [str(name) for name in active_branches if str(name).lower() not in {"", "none", "full", "all"}]
    return active or None


def pack_active_latent(z_b, branch_slices, active_z):
    parts = []
    for name, slc in branch_slices.items():
        if name in active_z:
            parts.append(active_z[name])
        else:
            parts.append(z_b[:, slc].detach())
    return torch.cat(parts, dim=1)


def branch_background_loss(active_z, z_b, branch_slices, background_error, reduction="mean", eps=1e-8):
    if isinstance(background_error, dict) and background_error.get("type") == "latent_channel_cov":
        items = background_error["items"]
        losses = []
        for name, z_branch in active_z.items():
            if name not in items:
                raise KeyError(f"Channel B_z does not contain branch {name!r}. Available={list(items)}")
            z_b_branch = z_b[:, branch_slices[name]].detach()
            loss = channel_background_loss(
                z_branch,
                z_b_branch,
                items[name],
                reduction="sum",
                eps=eps,
            )
            losses.append(loss)
        total = torch.stack(losses).sum()
        if reduction == "sum":
            return total
        if reduction == "mean":
            # Match channel_background_loss(..., reduction="mean"): average
            # over batch/spatial rows once, not once per active branch.
            first = next(iter(active_z.values()))
            denom = first.shape[0] * first.shape[2] * first.shape[3]
            return total / max(float(denom), 1.0)
        raise ValueError(f"Unsupported background loss reduction: {reduction}")
    losses = []
    for name, z_branch in active_z.items():
        slc = branch_slices[name]
        z_b_branch = z_b[:, slc].detach()
        if torch.is_tensor(background_error):
            denom = background_error[slc].to(device=z_branch.device, dtype=z_branch.dtype)
            while denom.ndim < z_branch.ndim:
                denom = denom.unsqueeze(0)
        else:
            denom = float(background_error)
        losses.append(0.5 * ((z_branch - z_b_branch) ** 2 / (denom + eps)))
    if not losses:
        raise ValueError("active_z must contain at least one branch.")
    flat = torch.cat([loss.reshape(-1) for loss in losses], dim=0)
    if reduction == "sum":
        return flat.sum()
    if reduction == "mean":
        return flat.mean()
    raise ValueError(f"Unsupported background loss reduction: {reduction}")


def branch_increment_logs(z, z_b, branch_slices):
    if not branch_slices:
        return {}
    logs = {}
    total = 0.0
    for name, slc in branch_slices.items():
        dz = z[:, slc] - z_b[:, slc]
        energy = float(torch.sum(dz * dz).detach().cpu().item())
        mean_energy = float(torch.mean(dz * dz).detach().cpu().item())
        logs[f"branch_increment_energy_{name}"] = energy
        logs[f"branch_increment_mean_energy_{name}"] = mean_energy
        total += energy
    logs["branch_increment_energy_total"] = total
    if total > 0:
        for name in branch_slices:
            logs[f"branch_increment_fraction_{name}"] = logs[f"branch_increment_energy_{name}"] / total
    else:
        for name in branch_slices:
            logs[f"branch_increment_fraction_{name}"] = 0.0
    return logs


def ccmp_grid_obs_loss(
    decoded,
    obs,
    obs_weight,
    obs_channel_indices,
    obs_error_std,
    obs_error_var=None,
    eps=1e-8,
):
    pred_obs = decoded[:, obs_channel_indices]
    if obs_error_var is None:
        denom_var = float(obs_error_std) ** 2
    else:
        denom_var = obs_error_var.to(device=decoded.device, dtype=decoded.dtype)
        while denom_var.ndim < pred_obs.ndim:
            denom_var = denom_var.unsqueeze(0)
    diff2 = (pred_obs - obs) ** 2 / (denom_var + eps)
    denom = obs_weight.sum().clamp_min(eps)
    return 0.5 * (diff2 * obs_weight).sum() / denom


def bilinear_sample_points(field, lat, lon, grid_lat, grid_lon):
    if lat.numel() == 0:
        return field.new_empty(0)

    grid_lat = torch.as_tensor(grid_lat, device=field.device, dtype=field.dtype).flatten()
    grid_lon = torch.as_tensor(grid_lon, device=field.device, dtype=field.dtype).flatten()
    lat = lat.to(device=field.device, dtype=field.dtype)
    lon = lon.to(device=field.device, dtype=field.dtype)

    lon_min = torch.min(grid_lon)
    lon_max = torch.max(grid_lon)
    lat_min = torch.min(grid_lat)
    lat_max = torch.max(grid_lat)
    x_norm = 2.0 * (lon - lon_min) / (lon_max - lon_min + 1e-12) - 1.0
    if grid_lat[0] <= grid_lat[-1]:
        y_norm = 2.0 * (lat - lat_min) / (lat_max - lat_min + 1e-12) - 1.0
    else:
        y_norm = 2.0 * (lat_max - lat) / (lat_max - lat_min + 1e-12) - 1.0

    sample_grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        field,
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.view(-1)


def icoads_t2m_obs_loss(
    decoded,
    icoads_obs,
    t2m_channel_index,
    grid_lat,
    grid_lon,
    t2m_error_std,
    t2m_std,
    time_r_mode="none",
    time_error_rate=0.0,
    time_scale_hours=1.0,
    eps=1e-8,
):
    if icoads_obs is None:
        zero = decoded.sum() * 0.0
        return zero, 0

    losses = []
    total_count = 0
    obs_var = float(t2m_error_std) ** 2
    t2m_std2 = (float(t2m_std) + 1e-6) ** 2
    time_r_mode = str(time_r_mode).lower()
    grid_lat_t = torch.as_tensor(grid_lat, device=decoded.device, dtype=decoded.dtype).flatten()
    grid_lon_t = torch.as_tensor(grid_lon, device=decoded.device, dtype=decoded.dtype).flatten()
    lat_min = torch.min(grid_lat_t)
    lat_max = torch.max(grid_lat_t)
    lon_min = torch.min(grid_lon_t)
    lon_max = torch.max(grid_lon_t)
    for bi, obs_item in enumerate(icoads_obs):
        n_obs = int(obs_item["value"].numel())
        if n_obs == 0:
            continue
        lat = obs_item["lat"].to(device=decoded.device, dtype=decoded.dtype)
        lon = obs_item["lon"].to(device=decoded.device, dtype=decoded.dtype)
        value = obs_item["value"].to(device=decoded.device, dtype=decoded.dtype)
        inside = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
        n_inside = int(inside.sum().detach().item())
        if n_inside == 0:
            continue
        lat = lat[inside]
        lon = lon[inside]
        value = value[inside]
        field = decoded[bi : bi + 1, t2m_channel_index : t2m_channel_index + 1]
        pred = bilinear_sample_points(
            field,
            lat,
            lon,
            grid_lat=grid_lat,
            grid_lon=grid_lon,
        )
        diff2 = (pred - value) ** 2
        if time_r_mode == "none":
            denom_var = obs_var / t2m_std2
        else:
            dt_hours = obs_item["dt_hours"].to(device=decoded.device, dtype=decoded.dtype)[inside]
            if time_r_mode == "linear":
                physical_var = obs_var + (float(time_error_rate) * dt_hours) ** 2
            elif time_r_mode == "gaussian":
                tau = max(float(time_scale_hours), 1e-6)
                physical_var = obs_var * torch.exp(0.5 * (dt_hours / tau) ** 2)
            else:
                raise ValueError(
                    f"Unsupported icoads time R mode {time_r_mode!r}; "
                    "use none, linear, or gaussian."
                )
            denom_var = physical_var / t2m_std2
        losses.append(0.5 * (diff2 / (denom_var + eps)).sum())
        total_count += n_inside

    if total_count == 0:
        zero = decoded.sum() * 0.0
        return zero, 0
    return torch.stack(losses).sum() / float(total_count), total_count


def latent_3dvar_adam(
    model,
    background,
    obs,
    obs_weight,
    obs_channel_indices,
    point_obs=None,
    point_obs_norm_stats=None,
    point_obs_target_vars=None,
    point_obs_reduction="mean",
    icoads_obs=None,
    icoads_t2m_channel_index=None,
    grid_lat=None,
    grid_lon=None,
    max_steps=50,
    lr=0.05,
    background_error=1.0,
    background_loss_reduction="mean",
    obs_error_std=0.5,
    obs_error_var=None,
    lambda_obs=1.0,
    lambda_icoads=1.0,
    icoads_t2m_error_std=2.0,
    t2m_std=1.0,
    icoads_time_r_mode="none",
    icoads_time_error_rate=0.0,
    icoads_time_scale_hours=1.0,
    grad_clip=1.0,
    active_branches=None,
):
    with torch.no_grad():
        z_b, output_size = encode_det_with_size(model, background)

    active_branches = normalize_active_branches(active_branches)
    branch_slices = branch_slices_from_model(model)
    if active_branches is not None:
        if branch_slices is None:
            raise ValueError("--active-branches requires a model with branch_latent.branch_channels.")
        unknown = sorted(set(active_branches) - set(branch_slices))
        if unknown:
            raise ValueError(f"Unknown active branches {unknown}; available={list(branch_slices)}")
        active_z = {
            name: z_b[:, branch_slices[name]].detach().clone().requires_grad_(True)
            for name in active_branches
        }
        opt_params = list(active_z.values())
        optimizer = torch.optim.Adam(opt_params, lr=float(lr))
        initial_z = pack_active_latent(z_b.detach(), branch_slices, active_z).detach()
    else:
        z = z_b.detach().clone().requires_grad_(True)
        active_z = None
        opt_params = [z]
        optimizer = torch.optim.Adam(opt_params, lr=float(lr))
        initial_z = z.detach().clone()

    best = {
        "loss": None,
        "z": initial_z,
        "decoded": None,
        "logs": {},
    }

    for step in range(int(max_steps)):
        optimizer.zero_grad(set_to_none=True)
        if active_z is not None:
            z = pack_active_latent(z_b.detach(), branch_slices, active_z)
            loss_b = branch_background_loss(
                active_z,
                z_b.detach(),
                branch_slices,
                background_error,
                reduction=background_loss_reduction,
            )
        else:
            loss_b = latent_background_loss(
                z,
                z_b.detach(),
                background_error,
                reduction=background_loss_reduction,
            )
        decoded = decode_det_with_size(model, z, output_size)
        if point_obs is None:
            loss_ccmp = ccmp_grid_obs_loss(
                decoded=decoded,
                obs=obs,
                obs_weight=obs_weight,
                obs_channel_indices=obs_channel_indices,
                obs_error_std=obs_error_std,
                obs_error_var=obs_error_var,
            )
            loss_point = decoded.sum() * 0.0
            point_diag = {"count": 0, "variable_counts": {}}
        else:
            loss_ccmp = decoded.sum() * 0.0
            loss_point, point_diag = point_observation_loss(
                state=decoded[:, : len(point_obs_target_vars)],
                observation_batches=point_obs,
                target_vars=point_obs_target_vars,
                norm_stats=point_obs_norm_stats,
                reduction=point_obs_reduction,
            )
        if icoads_obs is not None:
            loss_icoads, icoads_count = icoads_t2m_obs_loss(
                decoded=decoded,
                icoads_obs=icoads_obs,
                t2m_channel_index=icoads_t2m_channel_index,
                grid_lat=grid_lat,
                grid_lon=grid_lon,
                t2m_error_std=icoads_t2m_error_std,
                t2m_std=t2m_std,
                time_r_mode=icoads_time_r_mode,
                time_error_rate=icoads_time_error_rate,
                time_scale_hours=icoads_time_scale_hours,
            )
        else:
            loss_icoads = decoded.sum() * 0.0
            icoads_count = 0
        loss_o = loss_ccmp + loss_point + float(lambda_icoads) * loss_icoads
        loss = loss_b + float(lambda_obs) * loss_o
        loss.backward()
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(opt_params, float(grad_clip))
        optimizer.step()

        loss_value = float(loss.detach().item())
        if best["loss"] is None or loss_value < best["loss"]:
            best["loss"] = loss_value
            best["z"] = z.detach().clone()
            best["logs"] = {
                "best_step": step,
                "loss_total": loss_value,
                "loss_b": float(loss_b.detach().item()),
                "loss_o": float(loss_o.detach().item()),
                "loss_ccmp": float(loss_ccmp.detach().item()),
                "loss_point": float(loss_point.detach().item()),
                "point_obs_count": float(point_diag["count"]),
                "loss_icoads": float(loss_icoads.detach().item()),
                "icoads_t2m_obs_count": float(icoads_count),
                "icoads_t2m_error_std": float(icoads_t2m_error_std),
                "icoads_time_r_mode": str(icoads_time_r_mode),
                "icoads_time_error_rate": float(icoads_time_error_rate),
                "icoads_time_scale_hours": float(icoads_time_scale_hours),
            }
            for variable_name, variable_count in point_diag["variable_counts"].items():
                best["logs"][f"point_obs_count_{variable_name}"] = float(variable_count)

    with torch.no_grad():
        best["decoded"] = decode_det_with_size(model, best["z"], output_size).detach()
        if branch_slices is not None:
            best["logs"].update(branch_increment_logs(best["z"], z_b.detach(), branch_slices))
        best["logs"]["active_branches"] = "full" if active_branches is None else "+".join(active_branches)
    return best["decoded"], best["z"], best["logs"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/swim_ir_ae_era5_recon_compressed.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ccmp-grid-root", default="data/standardized/ccmp_grid")
    parser.add_argument(
        "--obs-source",
        default="ccmp",
        choices=["ccmp", "gdas_osse"],
        help="Use legacy CCMP grids or the common five-variable GDAS-OSSE point operator.",
    )
    parser.add_argument(
        "--point-obs-file",
        default="data/GDAS/OSSE_202301/gdas_osse_obs_202301_bohai_seed42.nc",
    )
    parser.add_argument("--point-obs-network", default="extended", choices=["main", "extended"])
    parser.add_argument("--point-obs-value-key", default="synthetic_obs")
    parser.add_argument("--point-obs-r-scale", type=float, default=1.0)
    parser.add_argument("--point-obs-reduction", default="mean", choices=["mean", "sum"])
    parser.add_argument(
        "--point-obs-variables",
        nargs="*",
        default=None,
        help=(
            "Optional model-variable subset, e.g. --point-obs-variables "
            "era5_u10 era5_v10. Omit to assimilate all five variables."
        ),
    )
    parser.add_argument("--use-icoads", action="store_true")
    parser.add_argument("--icoads-root", default="data/standardized/icoads_region")
    parser.add_argument("--icoads-window-hours", type=float, default=3.0)
    parser.add_argument("--icoads-t2m-error-std", type=float, default=2.0)
    parser.add_argument("--icoads-qc", default="strict", choices=["strict", "phys", "none"])
    parser.add_argument("--lambda-icoads", type=float, default=1.0)
    parser.add_argument(
        "--icoads-time-r-mode",
        default="none",
        choices=["none", "linear", "gaussian"],
        help=(
            "Temporal representativeness error model for ICOADS AT. "
            "none keeps a constant diagonal R; linear uses "
            "R=sigma^2+(rate*|dt|)^2; gaussian uses "
            "R=sigma^2*exp(0.5*(dt/tau)^2)."
        ),
    )
    parser.add_argument(
        "--icoads-time-error-rate",
        type=float,
        default=0.0,
        help="degC/hour rate used by --icoads-time-r-mode linear.",
    )
    parser.add_argument(
        "--icoads-time-scale-hours",
        type=float,
        default=1.0,
        help="tau in hours used by --icoads-time-r-mode gaussian.",
    )
    parser.add_argument(
        "--background-source",
        default="input",
        choices=["input", "gfs"],
        help=(
            "input uses cfg input_vars as the background. gfs uses batch raw_gfs "
            "physical fields and standardizes them with ERA5/target statistics "
            "before encoding by the AE."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--background-error-var", type=float, default=1.0)
    parser.add_argument("--latent-bz", default=None)
    parser.add_argument("--latent-bz-floor", type=float, default=1e-6)
    parser.add_argument("--latent-bz-scale", type=float, default=1.0)
    parser.add_argument("--background-loss-reduction", default="mean", choices=["mean", "sum"])
    parser.add_argument("--obs-error-std", type=float, default=0.5)
    parser.add_argument("--obs-r-diag", default=None)
    parser.add_argument("--obs-r-key", default="r_diag")
    parser.add_argument("--obs-r-floor", type=float, default=1e-6)
    parser.add_argument("--obs-r-ceiling", type=float, default=None)
    parser.add_argument("--obs-r-scale", type=float, default=1.0)
    parser.add_argument("--lambda-obs", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--active-branches",
        nargs="*",
        default=None,
        help=(
            "Selective branch 3DVar. Omit for full latent optimization. "
            "Examples: --active-branches wind ; --active-branches shared wind"
        ),
    )
    parser.add_argument("--use-ccmp-conf", action="store_true")
    parser.add_argument(
        "--missing-obs-policy",
        default="skip",
        choices=["skip", "error"],
        help=(
            "How to handle timestamps missing from ccmp_grid files. "
            "skip drops that DA sample; error raises immediately."
        ),
    )
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--fields-output",
        default=None,
        help=(
            "Optional NetCDF output containing physical-unit GFS background, "
            "AE background, analysis, and ERA5 reference fields."
        ),
    )
    args = parser.parse_args()

    if args.obs_source == "gdas_osse" and args.use_icoads:
        raise ValueError("--use-icoads cannot be combined with GDAS-OSSE because it duplicates surface observations.")

    cfg = load_config(args.config)
    cfg["batch_size"] = 1
    cfg["num_workers"] = 0
    cfg["persistent_workers"] = False

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = str(Path(cfg["exp_dir"]) / "checkpoints" / "best.pt")

    device = resolve_device(args.device)
    loader, dataset, _ = build_dataloader(cfg, split=args.split, distributed=False)
    model = load_model(cfg, checkpoint, device)
    latent_bz = load_latent_bz(
        args.latent_bz,
        device=device,
        floor=args.latent_bz_floor,
        scale=args.latent_bz_scale,
    )
    obs_r_diag = load_obs_r_diag(
        args.obs_r_diag,
        device=device,
        key=args.obs_r_key,
        floor=args.obs_r_floor,
        ceiling=args.obs_r_ceiling,
        scale=args.obs_r_scale,
    )

    target_vars = list(cfg["target_vars"])
    obs_channel_indices = [target_vars.index(CCMP_TO_TARGET[var]) for var in CCMP_TO_TARGET]
    t2m_channel_index = target_vars.index("era5_t2m") if "era5_t2m" in target_vars else None
    channel_names = list(cfg["target_vars"])
    if args.obs_source == "ccmp":
        obs_loader = CcmpGridObsLoader(
            root=args.ccmp_grid_root,
            norm_stats=dataset.norm_stats,
            target_vars=target_vars,
            use_conf_weight=args.use_ccmp_conf,
        )
    else:
        obs_loader = GdasOssePointObsLoader(
            path=args.point_obs_file,
            target_vars=target_vars,
            grid_latitude=dataset.lat,
            grid_longitude=dataset.lon,
            network=args.point_obs_network,
            value_key=args.point_obs_value_key,
            r_scale=args.point_obs_r_scale,
            variables=args.point_obs_variables,
        )
    icoads_loader = None
    if args.use_icoads:
        if t2m_channel_index is None:
            raise ValueError("--use-icoads requires 'era5_t2m' in cfg['target_vars'].")
        icoads_loader = IcoadsPointObsLoader(
            root=args.icoads_root,
            norm_stats=dataset.norm_stats,
            target_var="era5_t2m",
            window_hours=args.icoads_window_hours,
            qc=args.icoads_qc,
        )
    grid_lat = torch.as_tensor(dataset.lat, device=device, dtype=torch.float32)
    grid_lon = torch.as_tensor(dataset.lon, device=device, dtype=torch.float32)
    t2m_std = float(dataset.norm_stats.get("era5_t2m", {}).get("std", 1.0))
    rows = []
    field_store = {
        "gfs_background": [],
        "ae_background": [],
        "analysis": [],
        "era5_reference": [],
    }
    field_times = []
    skipped_missing_obs = 0
    try:
        for batch_i, batch in enumerate(tqdm(loader, desc="latent 3DVar", dynamic_ncols=True)):
            if args.num_batches > 0 and batch_i >= args.num_batches:
                break
            target = batch["target"].to(device, dtype=torch.float32, non_blocking=True)
            if args.background_source == "gfs":
                raw_gfs = batch["raw_gfs"].to(device, dtype=torch.float32, non_blocking=True)
                background_dynamic = standardize_raw_background_as_target(
                    raw_gfs,
                    norm_stats=dataset.norm_stats,
                    target_vars=target_vars,
                )
                background = append_static_channels_for_model(background_dynamic, batch, cfg)
            else:
                background = batch["input"].to(device, dtype=torch.float32, non_blocking=True)
                background_dynamic = dynamic_channels_for_metrics(background, cfg)
            try:
                loaded_obs = obs_loader.load_batch(batch["time"], device=device)
            except KeyError:
                if args.missing_obs_policy == "error":
                    raise
                skipped_missing_obs += 1
                continue
            if args.obs_source == "ccmp":
                obs, obs_weight = loaded_obs
                point_obs = None
                point_obs_count = 0
            else:
                obs = obs_weight = None
                point_obs = loaded_obs
                point_obs_count = sum(item.count for item in point_obs)
            icoads_obs = (
                icoads_loader.load_batch(batch["time"], device=device)
                if icoads_loader is not None
                else None
            )

            with torch.no_grad():
                background_recon = model(background)

            analysis, _, logs = latent_3dvar_adam(
                model=model,
                background=background,
                obs=obs,
                obs_weight=obs_weight,
                obs_channel_indices=obs_channel_indices,
                point_obs=point_obs,
                point_obs_norm_stats=dataset.norm_stats,
                point_obs_target_vars=target_vars,
                point_obs_reduction=args.point_obs_reduction,
                icoads_obs=icoads_obs,
                icoads_t2m_channel_index=t2m_channel_index,
                grid_lat=grid_lat,
                grid_lon=grid_lon,
                max_steps=args.max_steps,
                lr=args.lr,
                background_error=latent_bz if latent_bz is not None else args.background_error_var,
                background_loss_reduction=args.background_loss_reduction,
                obs_error_std=args.obs_error_std,
                obs_error_var=obs_r_diag,
                lambda_obs=args.lambda_obs,
                lambda_icoads=args.lambda_icoads,
                icoads_t2m_error_std=args.icoads_t2m_error_std,
                t2m_std=t2m_std,
                icoads_time_r_mode=args.icoads_time_r_mode,
                icoads_time_error_rate=args.icoads_time_error_rate,
                icoads_time_scale_hours=args.icoads_time_scale_hours,
                grad_clip=args.grad_clip,
                active_branches=args.active_branches,
            )

            row = {
                "batch_i": batch_i,
                "time": batch["time"][0],
                "obs_source": args.obs_source,
                "obs_count": (
                    float(obs_weight.sum().detach().cpu().item())
                    if obs_weight is not None
                    else float(point_obs_count)
                ),
                "ccmp_obs_count": (
                    float(obs_weight.sum().detach().cpu().item())
                    if obs_weight is not None
                    else 0.0
                ),
                "icoads_window_hours": float(args.icoads_window_hours),
                "icoads_qc": args.icoads_qc,
                "icoads_time_r_mode": args.icoads_time_r_mode,
                "icoads_time_error_rate": float(args.icoads_time_error_rate),
                "icoads_time_scale_hours": float(args.icoads_time_scale_hours),
            }
            row.update(logs)
            row.update(metric_dict(background_dynamic.float(), target.float(), "background_gfs", channel_names))
            row.update(metric_dict(background_recon.float(), target.float(), "background_ae", channel_names))
            row.update(metric_dict(analysis.float(), target.float(), "analysis", channel_names))
            rows.append(row)
            if args.fields_output is not None:
                standardized_fields = {
                    "gfs_background": background_dynamic,
                    "ae_background": dynamic_channels_for_metrics(
                        background_recon, cfg
                    ),
                    "analysis": dynamic_channels_for_metrics(analysis, cfg),
                    "era5_reference": target,
                }
                with torch.no_grad():
                    for field_name, values in standardized_fields.items():
                        physical = destandardize_dynamic_fields(
                            values,
                            norm_stats=dataset.norm_stats,
                            target_vars=target_vars,
                        )
                        field_store[field_name].append(
                            physical.detach().cpu().numpy()
                        )
                field_times.append(batch["time"][0])
    finally:
        obs_loader.close()
        if icoads_loader is not None:
            icoads_loader.close()

    df = pd.DataFrame(rows)
    output = args.output
    if output is None:
        output = str(Path(cfg["exp_dir"]) / f"latent_3dvar_{args.split}.csv")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    if args.fields_output is not None:
        if not field_times:
            raise RuntimeError("No samples were available for physical-field output.")
        save_physical_fields_netcdf(
            path=args.fields_output,
            field_store=field_store,
            times=field_times,
            lat=dataset.lat,
            lon=dataset.lon,
            target_vars=target_vars,
            metadata={
                "title": "Latent 3DVar physical fields",
                "config": str(args.config),
                "checkpoint": str(checkpoint),
                "split": str(args.split),
                "background_source": str(args.background_source),
                "active_branches": (
                    "full"
                    if args.active_branches is None
                    else "+".join(args.active_branches)
                ),
                "normalization_inverse": (
                    "ERA5 training mean/std from cfg norm_stats_path"
                ),
            },
        )

    summary = {
        "config": args.config,
        "checkpoint": checkpoint,
        "split": args.split,
        "background_source": args.background_source,
        "latent_bz": args.latent_bz,
        "obs_r_diag": args.obs_r_diag,
        "obs_r_key": args.obs_r_key,
        "obs_r_scale": args.obs_r_scale,
        "obs_source": args.obs_source,
        "point_obs_file": args.point_obs_file if args.obs_source == "gdas_osse" else None,
        "point_obs_network": args.point_obs_network if args.obs_source == "gdas_osse" else None,
        "point_obs_value_key": args.point_obs_value_key if args.obs_source == "gdas_osse" else None,
        "point_obs_r_scale": args.point_obs_r_scale if args.obs_source == "gdas_osse" else None,
        "point_obs_reduction": args.point_obs_reduction if args.obs_source == "gdas_osse" else None,
        "point_obs_variables": args.point_obs_variables if args.obs_source == "gdas_osse" else None,
        "active_branches": args.active_branches,
        "background_loss_reduction": args.background_loss_reduction,
        "output": output,
        "fields_output": args.fields_output,
        "rows": len(df),
        "skipped_missing_obs": skipped_missing_obs,
        "mean_metrics": df.mean(numeric_only=True).to_dict() if not df.empty else {},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
