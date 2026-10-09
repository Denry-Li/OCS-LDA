# dataset_bohai.py

import os
import json
import glob
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


OBS_TOKEN_COLUMNS = [
    "ccmp_u10",
    "ccmp_v10",
    "ccmp_ws",
    "ccmp_conf",
    "lat_norm",
    "lon_norm",
]


def obs_neighbor_filename(year: int, radius: int, max_obs_per_grid: int, conf_topk_weight: float) -> str:
    weight = f"{conf_topk_weight:.6g}".replace(".", "p").replace("-", "m")
    return f"ccmp_neighbors_{year}_r{radius}_k{max_obs_per_grid}_w{weight}.npz"


def build_dataloader(params, split: str, distributed: bool = False):
    data_format = str(params.get("data_format", "bohai_yearly")).lower()
    if data_format in {"simvp_daily_nc", "simvp_daily", "simvp_nc"}:
        dataset = SimVPDailyDataset(params=params, split=split)
    else:
        dataset = BohaiAssimDataset(params=params, split=split)

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=(split == "train"),
            drop_last=(split == "train"),
        )

    loader = DataLoader(
        dataset,
        batch_size=int(params["batch_size"]),
        shuffle=(split == "train" and sampler is None),
        sampler=sampler,
        num_workers=int(params.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
        persistent_workers=bool(params.get("persistent_workers", False)),
        collate_fn=bohai_collate_fn,
    )

    return loader, dataset, sampler


def _build_obs_neighbors_single(
    obs_yx: torch.Tensor,
    obs_conf: torch.Tensor,
    height: int,
    width: int,
    radius: int,
    max_obs_per_grid: int,
    conf_topk_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_grid = height * width
    neighbor_idx = torch.zeros(num_grid, max_obs_per_grid, dtype=torch.long)
    neighbor_mask = torch.zeros(num_grid, max_obs_per_grid, dtype=torch.bool)
    nobs = int(obs_yx.shape[0])
    if nobs == 0 or max_obs_per_grid <= 0:
        return neighbor_idx, neighbor_mask

    yx_int = obs_yx.round().to(torch.long)
    buckets: dict[tuple[int, int], list[int]] = {}
    for idx in range(nobs):
        y = int(yx_int[idx, 0].item())
        x = int(yx_int[idx, 1].item())
        if 0 <= y < height and 0 <= x < width:
            buckets.setdefault((y, x), []).append(idx)

    eps = 1e-6
    for y in range(height):
        for x in range(width):
            candidates = []
            for yy in range(max(0, y - radius), min(height, y + radius + 1)):
                for xx in range(max(0, x - radius), min(width, x + radius + 1)):
                    candidates.extend(buckets.get((yy, xx), []))
            if not candidates:
                continue

            cand = torch.tensor(candidates, dtype=torch.long)
            dy = obs_yx[cand, 0].float() - float(y)
            dx = obs_yx[cand, 1].float() - float(x)
            dist2 = dy * dy + dx * dx
            score = dist2 - float(conf_topk_weight) * torch.log(
                obs_conf[cand].float().clamp_min(eps)
            )
            order = torch.argsort(score)[:max_obs_per_grid]
            chosen = cand[order]
            grid_i = y * width + x
            k = int(chosen.numel())
            neighbor_idx[grid_i, :k] = chosen
            neighbor_mask[grid_i, :k] = True

    return neighbor_idx, neighbor_mask


def bohai_collate_fn(samples: list[dict]) -> dict:
    batch = {}
    first = samples[0]
    for key in first:
        if (
            key.startswith("obs_")
            or key.startswith("_obs_")
            or key in {"neighbor_idx", "neighbor_mask"}
        ):
            continue
        values = [sample[key] for sample in samples]
        if torch.is_tensor(values[0]):
            batch[key] = torch.stack(values, dim=0)
        else:
            batch[key] = values

    if "obs_feat" not in first:
        return batch

    batch_size = len(samples)
    _, height, width = first["input"].shape
    num_grid = height * width
    max_nobs = max(int(sample["obs_feat"].shape[0]) for sample in samples)
    obs_dim = int(first["obs_feat"].shape[1])
    max_obs_per_grid = int(first.get("_max_obs_per_grid", 16))

    obs_feat = torch.zeros(batch_size, max_nobs, obs_dim, dtype=torch.float32)
    obs_yx = torch.zeros(batch_size, max_nobs, 2, dtype=torch.float32)
    obs_conf = torch.zeros(batch_size, max_nobs, dtype=torch.float32)
    obs_mask = torch.zeros(batch_size, max_nobs, dtype=torch.bool)
    neighbor_idx = torch.zeros(
        batch_size, num_grid, max_obs_per_grid, dtype=torch.long
    )
    neighbor_mask = torch.zeros(
        batch_size, num_grid, max_obs_per_grid, dtype=torch.bool
    )

    for bi, sample in enumerate(samples):
        nobs = int(sample["obs_feat"].shape[0])
        if nobs > 0:
            obs_feat[bi, :nobs] = sample["obs_feat"]
            obs_yx[bi, :nobs] = sample["obs_yx"]
            obs_conf[bi, :nobs] = sample["obs_conf"]
            obs_mask[bi, :nobs] = True
        if "neighbor_idx" in sample and "neighbor_mask" in sample:
            neighbor_idx[bi] = sample["neighbor_idx"]
            neighbor_mask[bi] = sample["neighbor_mask"]

    batch.update(
        {
            "obs_feat": obs_feat,
            "obs_yx": obs_yx,
            "obs_conf": obs_conf,
            "obs_mask": obs_mask,
            "neighbor_idx": neighbor_idx,
            "neighbor_mask": neighbor_mask,
        }
    )
    return batch


class BohaiAssimDataset(Dataset):
    def __init__(self, params, split: str):
        self.params = params
        self.split = split

        self.samples_root = params["samples_root"]
        self.norm_stats_path = params["norm_stats_path"]

        self.input_vars = list(params["input_vars"])
        self.ccmp_vars = list(params.get("ccmp_vars", []))
        self.static_vars = list(params.get("static_vars", []))
        self.target_vars = list(params["target_vars"])
        self.use_obs_tokens = bool(params.get("use_obs_tokens", False))
        self.ccmp_token_root = params.get(
            "ccmp_token_root", "data/standardized/ccmp_tokens"
        )
        self.ccmp_grid_source = str(params.get("ccmp_grid_source", "sample")).lower()
        self.ccmp_lwg_root = params.get(
            "ccmp_lwg_root", "data/standardized/ccmp_lwg_grid"
        )
        if self.ccmp_grid_source not in {"sample", "lwg"}:
            raise ValueError(
                "ccmp_grid_source must be 'sample' or 'lwg', "
                f"got {self.ccmp_grid_source!r}."
            )
        self.obs_neighbor_mode = str(params.get("obs_neighbor_mode", "cache")).lower()
        self.obs_neighbor_root = params.get(
            "obs_neighbor_root", "data/standardized/ccmp_neighbors"
        )
        self.obs_radius = int(params.get("obs_radius", 2))
        self.max_obs_per_grid = int(params.get("max_obs_per_grid", 16))
        self.obs_conf_topk_weight = float(params.get("obs_conf_topk_weight", 0.5))

        self.all_input_vars = self.input_vars + self.ccmp_vars + self.static_vars

        self.years = self._get_years_for_split(split)
        self.norm_stats = self._load_norm_stats(self.norm_stats_path)

        self.files_paths = []
        self.ccmp_grid_paths = []
        for year in self.years:
            path = os.path.join(self.samples_root, f"samples_{year}.nc")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing sample file: {path}")
            self.files_paths.append(path)
            if self.ccmp_grid_source == "lwg":
                lwg_path = os.path.join(
                    self.ccmp_lwg_root,
                    f"ccmp_lwg_grid_{year}.nc",
                )
                if not os.path.exists(lwg_path):
                    raise FileNotFoundError(f"Missing CCMP LWG grid file: {lwg_path}")
                self.ccmp_grid_paths.append(lwg_path)

        self.files = [None for _ in self.files_paths]
        self.ccmp_grid_files = [None for _ in self.ccmp_grid_paths]
        self.index = self._build_index()
        self.token_tables = self._load_token_tables() if self.use_obs_tokens else {}
        self.obs_neighbor_cache = {}

        ds0 = xr.open_dataset(self.files_paths[0])
        self.lat = ds0["lat"].values.astype(np.float32)
        self.lon = ds0["lon"].values.astype(np.float32)
        self.img_h = len(self.lat)
        self.img_w = len(self.lon)
        ds0.close()
        self.precomputed_neighbors = (
            self._load_precomputed_neighbors()
            if self.use_obs_tokens and self.obs_neighbor_mode == "precomputed"
            else {}
        )

        logging.info(f"[{split}] years: {self.years}")
        logging.info(f"[{split}] files: {len(self.files_paths)}")
        logging.info(f"[{split}] samples: {len(self.index)}")
        logging.info(f"[{split}] grid: {self.img_h} x {self.img_w}")
        logging.info(f"[{split}] input_vars: {self.all_input_vars}")
        logging.info(f"[{split}] target_vars: {self.target_vars}")
        logging.info(f"[{split}] ccmp_grid_source: {self.ccmp_grid_source}")
        logging.info(f"[{split}] use_obs_tokens: {self.use_obs_tokens}")
        logging.info(f"[{split}] obs_neighbor_mode: {self.obs_neighbor_mode}")

    def _get_years_for_split(self, split: str) -> List[int]:
        if split == "train":
            return list(self.params["train_years"])
        if split == "valid":
            return list(self.params["valid_years"])
        if split == "test":
            return list(self.params["test_years"])
        raise ValueError(f"Unknown split: {split}")

    def _load_norm_stats(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return stats["variables"]

    def _build_index(self) -> List[Tuple[int, int]]:
        index = []
        for file_i, path in enumerate(self.files_paths):
            ds = xr.open_dataset(path)
            ntime = ds.sizes["time"]
            for time_i in range(ntime):
                index.append((file_i, time_i))
            ds.close()
        return index

    def _load_token_tables(self) -> dict[int, dict[str, pd.DataFrame]]:
        token_tables = {}
        for year in self.years:
            path = os.path.join(self.ccmp_token_root, f"ccmp_tokens_{year}.parquet")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing CCMP token file: {path}")
            cols = [
                "time",
                "lat_norm",
                "lon_norm",
                "ccmp_u10",
                "ccmp_v10",
                "ccmp_ws",
                "ccmp_conf",
                "nearest_gfs_y",
                "nearest_gfs_x",
            ]
            df = pd.read_parquet(path, columns=cols)
            df["time_key"] = pd.to_datetime(df["time"]).dt.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
            token_tables[year] = {
                key: part.reset_index(drop=True)
                for key, part in df.groupby("time_key", sort=False)
            }
        return token_tables

    def _load_precomputed_neighbors(self) -> dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        if self.obs_neighbor_mode not in {"cache", "precomputed"}:
            raise ValueError(
                "obs_neighbor_mode must be 'cache' or 'precomputed', "
                f"got {self.obs_neighbor_mode!r}."
            )
        tables = {}
        for year in self.years:
            path = os.path.join(
                self.obs_neighbor_root,
                obs_neighbor_filename(
                    year,
                    self.obs_radius,
                    self.max_obs_per_grid,
                    self.obs_conf_topk_weight,
                ),
            )
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing precomputed obs neighbor file: {path}. "
                    "Run precompute_obs_neighbors.py first or set "
                    "obs_neighbor_mode: cache."
                )
            data = np.load(path, allow_pickle=False)
            time_keys = data["time_keys"].astype(str)
            neighbor_idx = data["neighbor_idx"]
            neighbor_mask = data["neighbor_mask"]
            if neighbor_idx.shape[1:] != (
                self.img_h * self.img_w,
                self.max_obs_per_grid,
            ):
                raise ValueError(
                    f"Unexpected neighbor_idx shape in {path}: "
                    f"{neighbor_idx.shape}"
                )
            tables[year] = {
                time_key: (
                    torch.from_numpy(neighbor_idx[i].astype(np.int64, copy=False)),
                    torch.from_numpy(neighbor_mask[i].astype(bool, copy=False)),
                )
                for i, time_key in enumerate(time_keys)
            }
        return tables

    def _token_feature_array(self, token_df: pd.DataFrame) -> np.ndarray:
        if token_df.empty:
            return np.zeros((0, len(OBS_TOKEN_COLUMNS)), dtype=np.float32)
        feat = token_df[OBS_TOKEN_COLUMNS].to_numpy(dtype=np.float32)
        for ci, var in enumerate(["ccmp_u10", "ccmp_v10", "ccmp_ws"]):
            if var in self.norm_stats:
                mean = float(self.norm_stats[var]["mean"])
                std = float(self.norm_stats[var]["std"])
                feat[:, ci] = (feat[:, ci] - mean) / (std + 1e-6)
        feat = np.nan_to_num(feat, nan=0.0).astype(np.float32)
        return feat

    def _get_obs_tokens(self, ds, time_i: int, time_value: str) -> dict:
        time_value = str(time_value)
        year = int(pd.Timestamp(time_value).year)
        token_df = self.token_tables.get(year, {}).get(time_value)
        if token_df is None or token_df.empty:
            obs_feat = torch.zeros((0, len(OBS_TOKEN_COLUMNS)), dtype=torch.float32)
            obs_yx = torch.zeros((0, 2), dtype=torch.float32)
            obs_conf = torch.zeros((0,), dtype=torch.float32)
        else:
            obs_feat = torch.from_numpy(self._token_feature_array(token_df))
            obs_yx = torch.from_numpy(
                token_df[["nearest_gfs_y", "nearest_gfs_x"]]
                .to_numpy(dtype=np.float32)
            )
            obs_conf = torch.from_numpy(
                token_df["ccmp_conf"].to_numpy(dtype=np.float32)
            )
        if self.obs_neighbor_mode == "precomputed":
            try:
                neighbor_idx, neighbor_mask = self.precomputed_neighbors[year][time_value]
            except KeyError as exc:
                raise KeyError(
                    f"Missing precomputed obs neighbors for {time_value}"
                ) from exc
        else:
            cache_key = (
                time_value,
                self.obs_radius,
                self.max_obs_per_grid,
                round(self.obs_conf_topk_weight, 6),
            )
            if cache_key in self.obs_neighbor_cache:
                neighbor_idx, neighbor_mask = self.obs_neighbor_cache[cache_key]
            else:
                neighbor_idx, neighbor_mask = _build_obs_neighbors_single(
                    obs_yx=obs_yx,
                    obs_conf=obs_conf,
                    height=self.img_h,
                    width=self.img_w,
                    radius=self.obs_radius,
                    max_obs_per_grid=self.max_obs_per_grid,
                    conf_topk_weight=self.obs_conf_topk_weight,
                )
                self.obs_neighbor_cache[cache_key] = (neighbor_idx, neighbor_mask)
        return {
            "obs_feat": obs_feat,
            "obs_yx": obs_yx,
            "obs_conf": obs_conf,
            "neighbor_idx": neighbor_idx,
            "neighbor_mask": neighbor_mask,
            "_obs_radius": self.obs_radius,
            "_max_obs_per_grid": self.max_obs_per_grid,
            "_obs_conf_topk_weight": self.obs_conf_topk_weight,
        }

    def _open_file(self, file_i: int):
        if self.files[file_i] is None:
            self.files[file_i] = xr.open_dataset(self.files_paths[file_i])
        return self.files[file_i]

    def _open_ccmp_grid_file(self, file_i: int):
        if self.ccmp_grid_source == "sample":
            return self._open_file(file_i)
        if self.ccmp_grid_files[file_i] is None:
            self.ccmp_grid_files[file_i] = xr.open_dataset(self.ccmp_grid_paths[file_i])
        return self.ccmp_grid_files[file_i]

    def __len__(self):
        return len(self.index)

    def _standardize(self, arr: np.ndarray, var: str) -> np.ndarray:
        if var not in self.norm_stats:
            return arr.astype(np.float32)

        mean = float(self.norm_stats[var]["mean"])
        std = float(self.norm_stats[var]["std"])
        return ((arr - mean) / (std + 1e-6)).astype(np.float32)

    def _standardize_ccmp(
        self, arr: np.ndarray, var: str, mask: np.ndarray
    ) -> np.ndarray:
        """
        CCMP 无效位置保持为 0。
        有效位置做 mean/std 标准化。
        """
        out = np.zeros_like(arr, dtype=np.float32)

        if var not in self.norm_stats:
            out[mask > 0] = arr[mask > 0]
            return out

        mean = float(self.norm_stats[var]["mean"])
        std = float(self.norm_stats[var]["std"])

        valid = mask > 0
        out[valid] = (arr[valid] - mean) / (std + 1e-6)
        return out.astype(np.float32)

    def _read_dynamic_var(self, ds, var: str, time_i: int) -> np.ndarray:
        return ds[var].isel(time=time_i).values.astype(np.float32)

    def _read_static_var(self, ds, var: str) -> np.ndarray:
        return ds[var].values.astype(np.float32)

    def _zeros_grid(self) -> np.ndarray:
        return np.zeros((self.img_h, self.img_w), dtype=np.float32)

    def _ones_grid(self) -> np.ndarray:
        return np.ones((self.img_h, self.img_w), dtype=np.float32)

    def _read_dynamic_optional(
        self,
        ds,
        var: str,
        time_i: int,
        default: str = "zeros",
    ) -> np.ndarray:
        if var in ds:
            return self._read_dynamic_var(ds, var, time_i)
        if default == "ones":
            return self._ones_grid()
        return self._zeros_grid()

    def _read_static_optional(
        self,
        ds,
        var: str,
        default: str = "zeros",
    ) -> np.ndarray:
        if var in ds:
            return self._read_static_var(ds, var)
        if default == "ones":
            return self._ones_grid()
        return self._zeros_grid()

    def _build_ccmp_mask(
        self,
        ccmp_u10: np.ndarray,
        ccmp_v10: np.ndarray,
        ccmp_conf: np.ndarray | None = None,
    ) -> np.ndarray:
        mask = np.isfinite(ccmp_u10) & np.isfinite(ccmp_v10)
        if ccmp_conf is not None:
            mask = mask & np.isfinite(ccmp_conf) & (ccmp_conf > 0)
        return mask.astype(np.float32)

    def __getitem__(self, idx: int):
        file_i, time_i = self.index[idx]
        ds = self._open_file(file_i)
        ccmp_ds = self._open_ccmp_grid_file(file_i)

        input_arrays = []

        # 1. GFS dynamic input
        for var in self.input_vars:
            arr = self._read_dynamic_var(ds, var, time_i)
            arr = self._standardize(arr, var)
            input_arrays.append(arr)

        # 2. CCMP dynamic input
        ccmp_mask = None
        if "ccmp_mask" in ccmp_ds:
            ccmp_mask = self._read_dynamic_var(ccmp_ds, "ccmp_mask", time_i)
        elif "ccmp_u10" in ccmp_ds and "ccmp_v10" in ccmp_ds:
            ccmp_conf_for_mask = None
            if "ccmp_conf" in ccmp_ds:
                ccmp_conf_for_mask = self._read_dynamic_var(
                    ccmp_ds, "ccmp_conf", time_i
                )
            ccmp_mask = self._build_ccmp_mask(
                self._read_dynamic_var(ccmp_ds, "ccmp_u10", time_i),
                self._read_dynamic_var(ccmp_ds, "ccmp_v10", time_i),
                ccmp_conf_for_mask,
            )

        for var in self.ccmp_vars:
            if var == "ccmp_mask" and var not in ccmp_ds:
                arr = ccmp_mask if ccmp_mask is not None else self._zeros_grid()
            else:
                arr = self._read_dynamic_optional(ccmp_ds, var, time_i)

            if var in ["ccmp_u10", "ccmp_v10", "ccmp_ws"]:
                if ccmp_mask is not None:
                    arr = self._standardize_ccmp(arr, var, ccmp_mask)
                else:
                    arr = self._standardize(arr, var)

            elif var in ["ccmp_mask", "ccmp_conf"]:
                # already 0-1
                arr = arr.astype(np.float32)

            else:
                arr = self._standardize(arr, var)

            input_arrays.append(arr)

        # 3. static input
        for var in self.static_vars:
            arr = self._read_static_optional(ds, var)

            if var in [
                "land_sea_mask",
                "land_binary_mask",
                "signed_distance_to_coast_norm",
                "lat_norm",
                "lon_norm",
            ]:
                arr = arr.astype(np.float32)
            else:
                arr = self._standardize(arr, var)

            input_arrays.append(arr)

        inp = np.stack(input_arrays, axis=0).astype(np.float32)

        # 4. target increment
        target_arrays = []
        for var in self.target_vars:
            arr = self._read_dynamic_var(ds, var, time_i)
            arr = self._standardize(arr, var)
            target_arrays.append(arr)

        target = np.stack(target_arrays, axis=0).astype(np.float32)

        # 5. metadata
        time_value = np.datetime_as_string(
            ds["time"].isel(time=time_i).values, unit="s"
        )

        sample = {
            "input": torch.from_numpy(inp),
            "target": torch.from_numpy(target),
            "time": time_value,
        }
        if self.use_obs_tokens:
            sample.update(self._get_obs_tokens(ds, time_i, time_value))

        # Extra physical-space fields used by composite losses.
        raw_gfs_vars = ["gfs_t2m", "gfs_u10", "gfs_v10", "gfs_d2m", "gfs_sp"]
        raw_gfs = np.stack(
            [self._read_dynamic_optional(ds, var, time_i) for var in raw_gfs_vars],
            axis=0,
        ).astype(np.float32)
        raw_gfs = np.nan_to_num(raw_gfs, nan=0.0).astype(np.float32)

        raw_ccmp_u10 = self._read_dynamic_optional(ccmp_ds, "ccmp_u10", time_i)
        raw_ccmp_v10 = self._read_dynamic_optional(ccmp_ds, "ccmp_v10", time_i)
        raw_ccmp_conf = self._read_dynamic_optional(
            ccmp_ds, "ccmp_conf", time_i, "ones"
        )
        raw_ccmp_mask = ccmp_mask
        if raw_ccmp_mask is None:
            raw_ccmp_mask = self._build_ccmp_mask(
                raw_ccmp_u10,
                raw_ccmp_v10,
                raw_ccmp_conf,
            )

        raw_ccmp_u10 = np.nan_to_num(raw_ccmp_u10, nan=0.0).astype(np.float32)
        raw_ccmp_v10 = np.nan_to_num(raw_ccmp_v10, nan=0.0).astype(np.float32)
        raw_ccmp_conf = np.nan_to_num(raw_ccmp_conf, nan=0.0).astype(np.float32)
        raw_ccmp_mask = np.nan_to_num(raw_ccmp_mask, nan=0.0).astype(np.float32)

        sample.update(
            {
                "raw_gfs": torch.from_numpy(raw_gfs),
                "ccmp_u10": torch.from_numpy(raw_ccmp_u10),
                "ccmp_v10": torch.from_numpy(raw_ccmp_v10),
                "ccmp_mask": torch.from_numpy(raw_ccmp_mask),
                "ccmp_conf": torch.from_numpy(raw_ccmp_conf),
                "signed_distance_to_coast_km": torch.from_numpy(
                    self._read_static_optional(ds, "signed_distance_to_coast_km")
                ),
            }
        )

        return sample


class SimVPDailyDataset(Dataset):
    """
    Daily ERA5 metocean NetCDF adapter for SimVP-compatible AE training.

    Expected NetCDF layout:
        fields(time, channel, latitude, longitude)

    The AE reconstruction task uses the same normalized tensor for input and
    target, matching the batch interface consumed by train_bohai.py.
    """

    def __init__(self, params, split: str):
        self.params = params
        self.split = split
        self.data_path = params["data_path"]
        self.norm_stats_path = params["norm_stats_path"]
        self.channel_order_path = params.get("channel_order_path")
        self.input_vars = list(params["input_vars"])
        self.target_vars = list(params["target_vars"])

        if self.input_vars != self.target_vars:
            raise ValueError(
                "SimVPDailyDataset currently expects input_vars and target_vars "
                "to be identical for AE reconstruction."
            )
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Missing SimVP daily NetCDF: {self.data_path}")

        self.years = self._get_years_for_split(split)
        self.norm_stats = self._load_norm_stats(self.norm_stats_path)
        self.channel_order = self._load_channel_order()
        self.channel_indices = self._resolve_channel_indices()

        ds = xr.open_dataset(self.data_path)
        if "fields" not in ds:
            ds.close()
            raise KeyError(f"{self.data_path} must contain variable 'fields'.")
        expected_dims = ("time", "channel", "latitude", "longitude")
        if tuple(ds["fields"].dims) != expected_dims:
            actual = tuple(ds["fields"].dims)
            ds.close()
            raise ValueError(
                f"Unexpected fields dims in {self.data_path}: {actual}; "
                f"expected {expected_dims}."
            )

        self.times = pd.to_datetime(ds["time"].values)
        self.lat = ds["latitude"].values.astype(np.float32)
        self.lon = ds["longitude"].values.astype(np.float32)
        self.img_h = len(self.lat)
        self.img_w = len(self.lon)
        self.index = [
            time_i
            for time_i, timestamp in enumerate(self.times)
            if int(timestamp.year) in self.years
        ]
        ds.close()

        if not self.index:
            raise ValueError(f"No SimVP daily samples found for split={split}.")

        self.ds = None

        logging.info(f"[{split}] SimVP daily years: {self.years}")
        logging.info(f"[{split}] SimVP daily samples: {len(self.index)}")
        logging.info(f"[{split}] SimVP daily grid: {self.img_h} x {self.img_w}")
        logging.info(f"[{split}] SimVP daily variables: {self.input_vars}")

    def _get_years_for_split(self, split: str) -> List[int]:
        if split == "train":
            return list(self.params["train_years"])
        if split == "valid":
            return list(self.params["valid_years"])
        if split == "test":
            return list(self.params["test_years"])
        raise ValueError(f"Unknown split: {split}")

    def _load_norm_stats(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        if "variables" in stats:
            return stats["variables"]
        channels = list(stats["channels"])
        means = list(stats["mean"])
        stds = list(stats["std"])
        if not (len(channels) == len(means) == len(stds)):
            raise ValueError(f"Inconsistent SimVP stats lengths in {path}.")
        return {
            var: {"mean": float(mean), "std": float(std)}
            for var, mean, std in zip(channels, means, stds)
        }

    def _load_channel_order(self) -> list[str]:
        if self.channel_order_path and os.path.exists(self.channel_order_path):
            with open(self.channel_order_path, "r", encoding="utf-8") as f:
                channels = json.load(f)
            return [str(ch) for ch in channels]

        ds = xr.open_dataset(self.data_path)
        if "channel" not in ds.coords:
            ds.close()
            raise KeyError(
                "Missing channel_order_path and NetCDF has no channel coordinate."
            )
        channels = [str(ch) for ch in ds["channel"].values]
        ds.close()
        return channels

    def _resolve_channel_indices(self) -> list[int]:
        lookup = {name: idx for idx, name in enumerate(self.channel_order)}
        missing = [var for var in self.input_vars if var not in lookup]
        if missing:
            raise KeyError(
                f"Variables missing from SimVP channel order: {missing}. "
                f"Available: {self.channel_order}"
            )
        return [lookup[var] for var in self.input_vars]

    def _open_ds(self):
        if self.ds is None:
            self.ds = xr.open_dataset(self.data_path)
        return self.ds

    def __len__(self):
        return len(self.index)

    def _standardize_fields(self, arr: np.ndarray) -> np.ndarray:
        out = arr.astype(np.float32, copy=True)
        for ci, var in enumerate(self.input_vars):
            if var not in self.norm_stats:
                continue
            mean = float(self.norm_stats[var]["mean"])
            std = float(self.norm_stats[var]["std"])
            out[ci] = (out[ci] - mean) / (std + 1e-6)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def __getitem__(self, idx: int):
        time_i = self.index[idx]
        ds = self._open_ds()
        arr = (
            ds["fields"]
            .isel(time=time_i, channel=self.channel_indices)
            .values.astype(np.float32)
        )
        arr = self._standardize_fields(arr)
        time_value = np.datetime_as_string(
            ds["time"].isel(time=time_i).values, unit="s"
        )
        tensor = torch.from_numpy(arr)
        return {
            "input": tensor,
            "target": tensor.clone(),
            "time": time_value,
        }
