"""Shared differentiable point-observation operator for Bohai 3DVar methods.

The module deliberately separates the model-specific control-to-state mapping
from the common physical-space observation mapping.  Traditional grid-space
3DVar, LDA and PhySP_DA therefore use exactly the same interpolation weights,
physical units and diagonal observation-error variances.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import xarray as xr


def _axis_stencil(points: np.ndarray, coordinates: np.ndarray, axis_name: str):
    points = np.asarray(points, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64).reshape(-1)
    if coordinates.size < 2 or not np.isfinite(coordinates).all():
        raise ValueError(f"{axis_name} coordinates must contain at least two finite values")
    delta = np.diff(coordinates)
    ascending = bool(np.all(delta > 0))
    descending = bool(np.all(delta < 0))
    if not (ascending or descending):
        raise ValueError(f"{axis_name} coordinates must be strictly monotonic")
    if not np.isfinite(points).all():
        raise ValueError(f"Non-finite {axis_name} observation coordinate")

    ordered = coordinates if ascending else coordinates[::-1]
    scale = max(float(np.max(np.abs(ordered))), 1.0)
    tolerance = 32.0 * np.finfo(np.float64).eps * scale
    outside = (points < ordered[0] - tolerance) | (points > ordered[-1] + tolerance)
    if outside.any():
        bad = points[outside]
        raise ValueError(
            f"{bad.size} {axis_name} observations outside grid "
            f"[{ordered[0]}, {ordered[-1]}], examples={bad[:3].tolist()}"
        )
    clipped = np.clip(points, ordered[0], ordered[-1])
    upper = np.searchsorted(ordered, clipped, side="right")
    upper = np.clip(upper, 1, ordered.size - 1)
    lower = upper - 1
    denominator = ordered[upper] - ordered[lower]
    fraction = np.clip((clipped - ordered[lower]) / denominator, 0.0, 1.0)

    if ascending:
        index0, index1 = lower, upper
    else:
        index0 = coordinates.size - 1 - lower
        index1 = coordinates.size - 1 - upper
    return index0.astype(np.int64), index1.astype(np.int64), fraction.astype(np.float32)


def build_bilinear_stencil(
    latitude: np.ndarray,
    longitude: np.ndarray,
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
):
    """Build explicit four-corner indices and weights for a regional grid."""

    y0, y1, wy = _axis_stencil(latitude, grid_latitude, "latitude")
    x0, x1, wx = _axis_stencil(longitude, grid_longitude, "longitude")
    return {
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "wx": wx,
        "wy": wy,
    }


@dataclass(frozen=True)
class PointObservationBatch:
    """All observations belonging to one analysis cycle."""

    channel_index: torch.Tensor
    x0: torch.Tensor
    x1: torch.Tensor
    y0: torch.Tensor
    y1: torch.Tensor
    wx: torch.Tensor
    wy: torch.Tensor
    value: torch.Tensor
    error_variance: torch.Tensor
    variable_id: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.value.numel())


def sample_point_observations(state: torch.Tensor, observations: PointObservationBatch):
    """Apply H_point to one [C,H,W] state using explicit bilinear gathers."""

    if state.ndim != 3:
        raise ValueError(f"state must be [C,H,W], got {tuple(state.shape)}")
    if observations.count == 0:
        return state.new_empty(0)
    channel = observations.channel_index
    f00 = state[channel, observations.y0, observations.x0]
    f10 = state[channel, observations.y0, observations.x1]
    f01 = state[channel, observations.y1, observations.x0]
    f11 = state[channel, observations.y1, observations.x1]
    wx = observations.wx
    wy = observations.wy
    return (
        (1.0 - wx) * (1.0 - wy) * f00
        + wx * (1.0 - wy) * f10
        + (1.0 - wx) * wy * f01
        + wx * wy * f11
    )


def normalization_vectors(
    norm_stats: Mapping[str, Mapping[str, float]] | None,
    target_vars: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
):
    if norm_stats is None:
        mean = torch.zeros(len(target_vars), device=device, dtype=dtype)
        std = torch.ones(len(target_vars), device=device, dtype=dtype)
        return mean, std
    means, stds = [], []
    for name in target_vars:
        if name not in norm_stats:
            raise KeyError(f"Missing normalization statistics for {name!r}")
        means.append(float(norm_stats[name]["mean"]))
        # Training code uses std + 1e-6, so use the exact inverse here.
        stds.append(float(norm_stats[name]["std"]) + 1e-6)
    return (
        torch.tensor(means, device=device, dtype=dtype),
        torch.tensor(stds, device=device, dtype=dtype),
    )


def point_observation_loss(
    state: torch.Tensor,
    observation_batches: Sequence[PointObservationBatch],
    target_vars: Sequence[str],
    norm_stats: Mapping[str, Mapping[str, float]] | None = None,
    reduction: str = "mean",
    eps: float = 1e-8,
):
    """Compute 0.5 * innovation^T R^-1 innovation in physical units.

    ``state`` is normally standardized model output.  Passing ``norm_stats``
    converts sampled predictions back to physical units before applying R.
    With ``norm_stats=None``, state is interpreted as already physical.
    """

    if state.ndim != 4:
        raise ValueError(f"state must be [B,C,H,W], got {tuple(state.shape)}")
    if len(observation_batches) != state.shape[0]:
        raise ValueError(
            "One PointObservationBatch is required per state batch item: "
            f"{len(observation_batches)} != {state.shape[0]}"
        )
    reduction = str(reduction).lower()
    if reduction not in {"sum", "mean"}:
        raise ValueError(f"Unsupported point-observation reduction {reduction!r}")

    mean, std = normalization_vectors(norm_stats, target_vars, state.device, state.dtype)
    total = state.sum() * 0.0
    count = 0
    variable_counts = {name: 0 for name in target_vars}
    for batch_index, observations in enumerate(observation_batches):
        if observations.count == 0:
            continue
        prediction_standardized = sample_point_observations(state[batch_index], observations)
        prediction_physical = (
            prediction_standardized * std[observations.channel_index]
            + mean[observations.channel_index]
        )
        innovation = prediction_physical - observations.value
        total = total + 0.5 * torch.sum(
            innovation.square() / observations.error_variance.clamp_min(float(eps))
        )
        count += observations.count
        ids, counts = torch.unique(observations.channel_index, return_counts=True)
        for channel, channel_count in zip(ids.tolist(), counts.tolist()):
            variable_counts[target_vars[int(channel)]] += int(channel_count)
    if count == 0:
        return total, {"count": 0, "variable_counts": variable_counts}
    if reduction == "mean":
        total = total / float(count)
    return total, {"count": count, "variable_counts": variable_counts}


class GdasOssePointObsLoader:
    """Read formal GDAS-network OSSE cycles and cache a canonical H stencil."""

    def __init__(
        self,
        path: str | Path,
        target_vars: Sequence[str],
        grid_latitude: np.ndarray,
        grid_longitude: np.ndarray,
        network: str = "extended",
        value_key: str = "synthetic_obs",
        r_scale: float = 1.0,
        variables: Sequence[str] | None = None,
    ):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.target_vars = list(target_vars)
        self.network = str(network).lower()
        if self.network not in {"main", "extended"}:
            raise ValueError("network must be 'main' or 'extended'")
        self.value_key = str(value_key)
        self.r_scale = float(r_scale)
        if self.r_scale <= 0:
            raise ValueError("r_scale must be positive")
        self.dataset = xr.open_dataset(self.path)
        required = {
            self.value_key,
            "observation_error_variance",
            "cycle_index",
            "cycle_time",
            "latitude",
            "longitude",
            "model_variable",
            "variable_id",
            f"use_{self.network}",
        }
        missing = sorted(required - set(self.dataset.variables))
        if missing:
            self.close()
            raise KeyError(f"Missing point-observation variables: {missing}")

        stored_lat = np.asarray(self.dataset.target_latitude.values, dtype=np.float64)
        stored_lon = np.asarray(self.dataset.target_longitude.values, dtype=np.float64)
        grid_latitude = np.asarray(grid_latitude, dtype=np.float64)
        grid_longitude = np.asarray(grid_longitude, dtype=np.float64)
        if stored_lat.shape != grid_latitude.shape or not np.allclose(stored_lat, grid_latitude):
            self.close()
            raise ValueError("OSSE and model latitude grids differ")
        if stored_lon.shape != grid_longitude.shape or not np.allclose(stored_lon, grid_longitude):
            self.close()
            raise ValueError("OSSE and model longitude grids differ")

        if "model_variable_name" in self.dataset and "variable_id" in self.dataset:
            variable_names = self.dataset.model_variable_name.values.astype(str)
            variable_id = self.dataset.variable_id.values.astype(np.int64)
            if np.any(variable_id < 0) or np.any(variable_id >= variable_names.size):
                self.close()
                raise ValueError("OSSE variable_id is outside model_variable_name")
            model_variable = variable_names[variable_id]
        else:
            model_variable = self.dataset.model_variable.values.astype(str)
        unknown = sorted(set(model_variable) - set(self.target_vars))
        if unknown:
            self.close()
            raise ValueError(f"OSSE variables absent from model target_vars: {unknown}")
        lookup = {name: index for index, name in enumerate(self.target_vars)}
        self.channel_index = np.asarray([lookup[name] for name in model_variable], dtype=np.int64)
        self.variables = None if variables is None else list(variables)
        if self.variables is not None:
            unknown_requested = sorted(set(self.variables) - set(self.target_vars))
            if unknown_requested:
                self.close()
                raise ValueError(
                    f"Requested point-observation variables absent from target_vars: {unknown_requested}"
                )
            variable_selection = np.isin(model_variable, self.variables)
        else:
            variable_selection = np.ones(model_variable.shape, dtype=bool)
        self.variable_id = self.dataset.variable_id.values.astype(np.int64)
        self.value = self.dataset[self.value_key].values.astype(np.float32)
        self.error_variance = (
            self.dataset.observation_error_variance.values.astype(np.float32) * self.r_scale
        )
        if not np.isfinite(self.value).all() or not np.isfinite(self.error_variance).all():
            self.close()
            raise ValueError("Point observations and R must be finite")
        if np.any(self.error_variance <= 0):
            self.close()
            raise ValueError("Point-observation variances must be positive")
        self.use = (
            self.dataset[f"use_{self.network}"].values.astype(bool)
            & variable_selection
        )
        self.cycle_index = self.dataset.cycle_index.values.astype(np.int64)
        self.stencil = build_bilinear_stencil(
            self.dataset.latitude.values,
            self.dataset.longitude.values,
            grid_latitude,
            grid_longitude,
        )
        cycle_times = self.dataset.cycle_time.values.astype("datetime64[ns]")
        self.time_to_cycle = {
            np.datetime_as_string(value, unit="s"): index
            for index, value in enumerate(cycle_times)
        }
        if "cycle_obs_start" in self.dataset and "cycle_obs_count" in self.dataset:
            starts = self.dataset.cycle_obs_start.values.astype(np.int64)
            counts = self.dataset.cycle_obs_count.values.astype(np.int64)
            if starts.size != cycle_times.size or counts.size != cycle_times.size:
                self.close()
                raise ValueError("OSSE ragged cycle index has the wrong length")
            if int(counts.sum()) != self.cycle_index.size:
                self.close()
                raise ValueError("OSSE cycle_obs_count does not cover all observations")
            expected_starts = np.concatenate(([0], np.cumsum(counts[:-1])))
            if not np.array_equal(starts, expected_starts):
                self.close()
                raise ValueError("OSSE cycle_obs_start is not contiguous")
            self.cycle_indices = []
            for cycle, (start, count) in enumerate(zip(starts, counts)):
                indices = np.arange(start, start + count, dtype=np.int64)
                if not np.all(self.cycle_index[indices] == cycle):
                    self.close()
                    raise ValueError(f"OSSE cycle_index disagrees with ragged slice at cycle {cycle}")
                self.cycle_indices.append(indices[self.use[indices]])
        else:
            self.cycle_indices = [
                np.flatnonzero((self.cycle_index == index) & self.use)
                for index in range(len(cycle_times))
            ]

    def close(self):
        dataset = getattr(self, "dataset", None)
        if dataset is not None:
            dataset.close()
            self.dataset = None

    def _tensor(self, values, device, dtype):
        return torch.as_tensor(values, device=device, dtype=dtype)

    def load_batch(self, times, device, dtype=torch.float32):
        batches = []
        for time_value in times:
            time_key = pd.Timestamp(str(time_value)).strftime("%Y-%m-%dT%H:%M:%S")
            if time_key not in self.time_to_cycle:
                raise KeyError(f"{time_key} not found in {self.path}")
            indices = self.cycle_indices[self.time_to_cycle[time_key]]
            batches.append(
                PointObservationBatch(
                    channel_index=self._tensor(self.channel_index[indices], device, torch.long),
                    x0=self._tensor(self.stencil["x0"][indices], device, torch.long),
                    x1=self._tensor(self.stencil["x1"][indices], device, torch.long),
                    y0=self._tensor(self.stencil["y0"][indices], device, torch.long),
                    y1=self._tensor(self.stencil["y1"][indices], device, torch.long),
                    wx=self._tensor(self.stencil["wx"][indices], device, dtype),
                    wy=self._tensor(self.stencil["wy"][indices], device, dtype),
                    value=self._tensor(self.value[indices], device, dtype),
                    error_variance=self._tensor(self.error_variance[indices], device, dtype),
                    variable_id=self._tensor(self.variable_id[indices], device, torch.long),
                )
            )
        return batches
