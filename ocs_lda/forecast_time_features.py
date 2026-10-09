from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_time_features(
    valid_time: Any,
    lat: np.ndarray,
    lon: np.ndarray,
    mode: str = "local_solar",
) -> np.ndarray:
    """Build cyclic target-valid-time features with shape [4, H, W]."""

    timestamp = pd.Timestamp(str(valid_time))
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)

    lat = np.asarray(lat, dtype=np.float32)
    lon = np.asarray(lon, dtype=np.float32)
    height = int(lat.size)
    width = int(lon.size)

    utc_hour = (
        float(timestamp.hour)
        + float(timestamp.minute) / 60.0
        + float(timestamp.second) / 3600.0
    )
    if mode == "local_solar":
        local_hour = utc_hour + lon / 15.0
    elif mode == "utc":
        local_hour = np.full_like(lon, utc_hour, dtype=np.float32)
    elif mode == "beijing":
        local_hour = np.full_like(lon, utc_hour + 8.0, dtype=np.float32)
    else:
        raise ValueError(f"Unknown forecast_time_feature_mode: {mode}")

    daily_phase = 2.0 * np.pi * np.mod(local_hour, 24.0) / 24.0
    daily_sin = np.repeat(np.sin(daily_phase)[None, :], height, axis=0)
    daily_cos = np.repeat(np.cos(daily_phase)[None, :], height, axis=0)

    year_start = pd.Timestamp(year=timestamp.year, month=1, day=1)
    next_year = pd.Timestamp(year=timestamp.year + 1, month=1, day=1)
    year_fraction = (timestamp - year_start).total_seconds() / (
        next_year - year_start
    ).total_seconds()
    annual_phase = 2.0 * np.pi * year_fraction
    annual_sin = np.full((height, width), np.sin(annual_phase), dtype=np.float32)
    annual_cos = np.full((height, width), np.cos(annual_phase), dtype=np.float32)

    return np.stack(
        [daily_sin, daily_cos, annual_sin, annual_cos],
        axis=0,
    ).astype(np.float32)
