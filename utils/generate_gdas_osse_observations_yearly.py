#!/usr/bin/env python3
"""Generate deterministic full-year OSSE observations on an idealized GDAS network."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EXPECTED_VARIABLES = ["PRES", "TMP", "UGRD", "VGRD", "DPT"]
MODEL_VARIABLE = {
    "PRES": "era5_sp",
    "TMP": "era5_t2m",
    "UGRD": "era5_u10",
    "VGRD": "era5_v10",
    "DPT": "era5_d2m",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True, type=Path)
    parser.add_argument("--era5", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-dir", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_obs_ids(network: xr.Dataset) -> np.ndarray:
    cycle_ns = network.cycle_time.values.astype("datetime64[ns]").astype(np.int64)
    cycle_index = network.cycle_index.values.astype(np.int64)
    variables = network.source_variable.values.astype(str)
    platforms = network.platform.values.astype(str)
    station_keys = network.station_key.values.astype(str)
    record_kinds = network.record_kind.values.astype(str)
    source_indices = network.source_observation_index.values.astype(np.int64)
    ids = np.empty(network.sizes["obs"], dtype="U28")
    for index in range(network.sizes["obs"]):
        payload = "|".join(
            [
                str(int(cycle_ns[cycle_index[index]])),
                variables[index],
                platforms[index],
                station_keys[index],
                str(int(source_indices[index])),
                record_kinds[index],
            ]
        )
        ids[index] = "obs_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    if np.unique(ids).size != ids.size:
        raise RuntimeError("Stable observation ID collision")
    return ids


def stable_standard_normal(obs_ids: np.ndarray, seed: int) -> np.ndarray:
    output = np.empty(obs_ids.size, dtype=np.float32)
    denominator = float(2**64)
    for index, obs_id in enumerate(obs_ids):
        digest = hashlib.sha256(f"{seed}|{obs_id}".encode("utf-8")).digest()
        u1 = (int.from_bytes(digest[:8], "big") + 0.5) / denominator
        u2 = (int.from_bytes(digest[8:16], "big") + 0.5) / denominator
        output[index] = math.sqrt(-2.0 * math.log(u1)) * math.cos(math.tau * u2)
    return output


def bilinear_sample(field: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    height, width = field.shape
    x0 = np.clip(np.floor(grid_x).astype(np.int64), 0, width - 1)
    y0 = np.clip(np.floor(grid_y).astype(np.int64), 0, height - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = np.clip(grid_x - x0, 0.0, 1.0)
    wy = np.clip(grid_y - y0, 0.0, 1.0)
    f00 = field[y0, x0].astype(np.float64)
    f10 = field[y0, x1].astype(np.float64)
    f01 = field[y1, x0].astype(np.float64)
    f11 = field[y1, x1].astype(np.float64)
    sampled = (1 - wx) * (1 - wy) * f00 + wx * (1 - wy) * f10 + (1 - wx) * wy * f01 + wx * wy * f11
    return sampled, {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "wx": wx, "wy": wy, "f00": f00, "f10": f10, "f01": f01, "f11": f11}


def sample_nature_run(network: xr.Dataset, era5: xr.Dataset) -> tuple[np.ndarray, list[dict[str, object]]]:
    cycle_times = network.cycle_time.values.astype("datetime64[ns]")
    era5_times = era5.time.values.astype("datetime64[ns]")
    missing = cycle_times[~np.isin(cycle_times, era5_times)]
    if missing.size:
        raise ValueError(f"ERA5 is missing network cycles: {missing}")
    if not np.allclose(network.target_latitude.values, era5.lat.values) or not np.allclose(network.target_longitude.values, era5.lon.values):
        raise ValueError("ERA5 and network target coordinates differ")
    time_lookup = {value: index for index, value in enumerate(era5_times)}
    era5_fields = {name: era5[name].values.astype(np.float32) for name in MODEL_VARIABLE.values()}
    truth = np.full(network.sizes["obs"], np.nan, dtype=np.float32)
    variables = network.source_variable.values.astype(str)
    grid_x = network.grid_x.values.astype(np.float64)
    grid_y = network.grid_y.values.astype(np.float64)
    starts = network.cycle_obs_start.values.astype(np.int64)
    counts = network.cycle_obs_count.values.astype(np.int64)
    audit_cycles = set(np.unique(np.linspace(0, network.sizes["cycle"] - 1, 8, dtype=int)).tolist())
    corner_rows: list[dict[str, object]] = []
    for cycle, (start, count) in enumerate(zip(starts, counts)):
        cycle_indices = np.arange(start, start + count, dtype=np.int64)
        time_index = time_lookup[cycle_times[cycle]]
        for source_variable in EXPECTED_VARIABLES:
            indices = cycle_indices[variables[cycle_indices] == source_variable]
            if indices.size == 0:
                raise RuntimeError(f"No {source_variable} observations at cycle {cycle}")
            field = era5_fields[MODEL_VARIABLE[source_variable]][time_index]
            if field.shape != (40, 40) or not np.isfinite(field).all():
                raise ValueError(f"Invalid ERA5 field {source_variable} at {cycle_times[cycle]}")
            sampled, corners = bilinear_sample(field, grid_x[indices], grid_y[indices])
            truth[indices] = sampled.astype(np.float32)
            if cycle in audit_cycles:
                for local_index, global_index in enumerate(indices[:2]):
                    row = {
                        "obs_index": int(global_index),
                        "cycle_index": cycle,
                        "cycle_time": str(cycle_times[cycle]),
                        "variable": source_variable,
                        "truth": float(sampled[local_index]),
                    }
                    row.update({name: float(value[local_index]) for name, value in corners.items()})
                    corner_rows.append(row)
    if not np.isfinite(truth).all():
        raise RuntimeError("Non-finite bilinear truth samples")
    return truth, corner_rows


def build_output(network: xr.Dataset, truth: np.ndarray, obs_ids: np.ndarray, z: np.ndarray, seed: int, network_path: Path, era5_path: Path) -> xr.Dataset:
    output = network.drop_vars(["gdas_source_value", "gdas_aligned_value"], errors="ignore").copy(deep=True)
    error_std = output.observation_error_std.values.astype(np.float32)
    noise = (z * error_std).astype(np.float32)
    synthetic = (truth + noise).astype(np.float32)
    cycle_index = output.cycle_index.values.astype(np.int64)
    nature_time = output.cycle_time.values[cycle_index].astype("datetime64[ns]")
    output["obs_id"] = ("obs", obs_ids)
    output["nature_run_time"] = ("obs", nature_time)
    output["truth_at_obs"] = ("obs", truth.astype(np.float32))
    output["noise_standard_normal"] = ("obs", z.astype(np.float32))
    output["noise"] = ("obs", noise)
    output["synthetic_obs"] = ("obs", synthetic)
    output["observation_error_variance"] = ("obs", np.square(error_std).astype(np.float32))
    cycle_times = output.cycle_time.values.astype("datetime64[ns]")
    period_label = f"{str(cycle_times[0])[:10]} to {str(cycle_times[-1])[:10]}"
    output.attrs.update(
        {
            "title": f"Formal real-network-driven idealized Bohai-Yellow Sea OSSE observations ({period_label})",
            "source_network": str(network_path),
            "source_network_sha256": sha256_file(network_path),
            "nature_run": str(era5_path),
            "cycles": f"{cycle_times.size} six-hourly cycles from {cycle_times[0]} through {cycle_times[-1]}",
            "sampling_mode": "synchronous_cycle_time",
            "spatial_operator": "bilinear interpolation on ascending 40x40 0.25-degree ERA5 grid",
            "noise_seed": int(seed),
            "noise_distribution": "independent Gaussian N(0, observation_error_std^2)",
            "noise_generator": "SHA-256(seed|obs_id) uniforms with Box-Muller; independent of row/batch order",
            "formal_observation_variable": "synthetic_obs",
            "real_gdas_values_removed": "true",
            "osse_note": "Assimilate synthetic_obs only; truth_at_obs is verification truth.",
            "history": f"created {datetime.now(timezone.utc).isoformat()} by generate_gdas_osse_observations_yearly.py",
        }
    )
    return output


def validate(network: xr.Dataset, output: xr.Dataset, truth: np.ndarray, obs_ids: np.ndarray, z: np.ndarray, seed: int) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []

    def add(name: str, condition: bool, detail: object) -> None:
        checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": str(detail)})

    add("dimensions_preserved", dict(output.sizes) == dict(network.sizes), dict(output.sizes))
    add(
        "cycle_count_matches_network",
        output.sizes["cycle"] == network.sizes["cycle"] and output.sizes["cycle"] > 0,
        output.sizes["cycle"],
    )
    add("real_gdas_values_absent", "gdas_source_value" not in output and "gdas_aligned_value" not in output, "formal OSSE contains no real GDAS values")
    core = np.column_stack([output.truth_at_obs.values, output.noise_standard_normal.values, output.noise.values, output.synthetic_obs.values, output.observation_error_variance.values])
    add("finite_osse_values", bool(np.isfinite(core).all()), "truth,z,noise,synthetic,R")
    relation = (output.truth_at_obs.values + output.noise.values).astype(np.float32)
    add("synthetic_identity", np.array_equal(output.synthetic_obs.values, relation), float(np.max(np.abs(output.synthetic_obs.values - relation))))
    variance = np.square(output.observation_error_std.values.astype(np.float64))
    add("variance_identity", np.allclose(output.observation_error_variance.values, variance, atol=1e-6), float(np.max(np.abs(output.observation_error_variance.values - variance))))
    add("truth_roundtrip", np.array_equal(output.truth_at_obs.values, truth.astype(np.float32)), "exact float32")
    add("obs_id_unique", np.unique(output.obs_id.values.astype(str)).size == output.sizes["obs"], output.sizes["obs"])
    sample = np.linspace(0, obs_ids.size - 1, min(20000, obs_ids.size), dtype=np.int64)
    regenerated = stable_standard_normal(obs_ids[sample], seed)
    add("noise_reproducibility_sample", np.array_equal(regenerated, output.noise_standard_normal.values[sample]), f"sample={sample.size}")
    cycle_index = output.cycle_index.values.astype(np.int64)
    add("synchronous_sampling_time", np.array_equal(output.nature_run_time.values, output.cycle_time.values[cycle_index]), f"{output.nature_run_time.values[0]}..{output.nature_run_time.values[-1]}")
    z_mean = float(z.mean())
    z_std = float(z.std(ddof=0))
    mean_tolerance = max(0.01, 4.0 / math.sqrt(z.size))
    std_tolerance = max(0.01, 4.0 / math.sqrt(2.0 * z.size))
    add(
        "standard_normal_global",
        abs(z_mean) < mean_tolerance and abs(z_std - 1.0) < std_tolerance,
        (
            f"mean={z_mean:.6f}, std={z_std:.6f}, "
            f"mean_tol={mean_tolerance:.6f}, std_tol={std_tolerance:.6f}"
        ),
    )
    template_size = int(network.attrs.get("source_template_observations", 0))
    if template_size <= 0 and "template_observation_index" in network:
        template_size = int(network.template_observation_index.values.max()) + 1
    if 0 < template_size and z.size >= 2 * template_size:
        repeat_corr = float(np.corrcoef(z[:template_size], z[template_size : 2 * template_size])[0, 1])
        add("repeated_geometry_noise_independence", abs(repeat_corr) < 0.02, f"corr={repeat_corr:.6f}")
    if any(item["status"] == "FAIL" for item in checks):
        raise RuntimeError(f"OSSE validation failed: {[item for item in checks if item['status'] == 'FAIL']}")
    return checks


def write_audit(output: xr.Dataset, audit_dir: Path, checks: list[dict[str, object]], corner_rows: list[dict[str, object]]) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    variables = output.source_variable.values.astype(str)
    platforms = output.platform.values.astype(str)
    cycle_index = output.cycle_index.values.astype(np.int64)
    z = output.noise_standard_normal.values.astype(np.float64)
    truth = output.truth_at_obs.values.astype(np.float64)
    synthetic = output.synthetic_obs.values.astype(np.float64)
    frame = pd.DataFrame(
        {
            "variable": variables,
            "platform": platforms,
            "cycle_index": cycle_index,
            "main": output.use_main.values.astype(bool),
            "z": z,
            "truth": truth,
            "synthetic": synthetic,
            "error_std": output.observation_error_std.values,
        }
    )
    summary = frame.groupby(["platform", "variable"], observed=True).agg(observations=("z", "size"), main=("main", "sum"), z_mean=("z", "mean"), z_std=("z", lambda value: value.std(ddof=0)), truth_min=("truth", "min"), truth_max=("truth", "max"), error_std_mean=("error_std", "mean")).reset_index()
    summary.to_csv(audit_dir / "platform_variable_noise_summary.csv", index=False)
    cycle_times = output.cycle_time.values.astype("datetime64[ns]")
    monthly = frame.assign(month=cycle_times[cycle_index].astype("datetime64[M]").astype(str)).groupby(["month", "variable"], observed=True).agg(observations=("z", "size"), z_mean=("z", "mean"), z_std=("z", lambda value: value.std(ddof=0)), truth_min=("truth", "min"), truth_max=("truth", "max")).reset_index()
    monthly.to_csv(audit_dir / "monthly_variable_summary.csv", index=False)
    pd.DataFrame(checks).to_csv(audit_dir / "generation_validation_checks.csv", index=False)
    pd.DataFrame(corner_rows).to_csv(audit_dir / "bilinear_corner_audit.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bins = np.linspace(-4.5, 4.5, 91)
    ax.hist(z, bins=bins, density=True, alpha=0.65, label="Generated z")
    x = np.linspace(-4.5, 4.5, 500)
    ax.plot(x, np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi), color="black", linewidth=1.4, label="N(0,1)")
    ax.set(
        title=f"Deterministic OSSE noise (seed {int(output.attrs['noise_seed'])})",
        xlabel="Standardized noise",
        ylabel="Density",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(audit_dir / "standardized_noise_histogram.png", dpi=180)
    plt.close(fig)

    sample = np.linspace(0, truth.size - 1, min(30000, truth.size), dtype=np.int64)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, variable in zip(axes.flat, EXPECTED_VARIABLES):
        selected = sample[variables[sample] == variable]
        ax.scatter(truth[selected], synthetic[selected], s=4, alpha=0.15)
        low = min(float(truth[selected].min()), float(synthetic[selected].min()))
        high = max(float(truth[selected].max()), float(synthetic[selected].max()))
        ax.plot([low, high], [low, high], color="black", linewidth=1)
        ax.set(title=variable, xlabel="ERA5 truth", ylabel="Synthetic observation")
        ax.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    fig.suptitle("Full-year truth versus synthetic observations")
    fig.savefig(audit_dir / "truth_vs_synthetic_scatter.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(args.network) as opened:
        network = opened.load()
    with xr.open_dataset(args.era5) as opened:
        era5 = opened.load()
    truth, corner_rows = sample_nature_run(network, era5)
    obs_ids = make_obs_ids(network)
    z = stable_standard_normal(obs_ids, args.seed)
    output = build_output(network, truth, obs_ids, z, args.seed, args.network, args.era5)
    checks = validate(network, output, truth, obs_ids, z, args.seed)
    write_audit(output, args.audit_dir, checks, corner_rows)
    encoding = {
        name: {"zlib": True, "complevel": 4, "shuffle": True}
        for name, variable in output.data_vars.items()
        if variable.dtype.kind not in {"O", "U", "M"}
    }
    output.to_netcdf(args.output, engine="netcdf4", encoding=encoding)
    checksum = sha256_file(args.output)
    with xr.open_dataset(args.output) as reopened:
        reopened_checks = validate(network, reopened, truth, obs_ids, z, args.seed)
    if reopened_checks != checks:
        raise RuntimeError("In-memory and reopened validation differ")
    summary = {
        "network": str(args.network),
        "network_sha256": sha256_file(args.network),
        "era5": str(args.era5),
        "output": str(args.output),
        "output_sha256": checksum,
        "seed": args.seed,
        "cycles": int(output.sizes["cycle"]),
        "observations": int(output.sizes["obs"]),
        "standard_normal_mean": float(z.mean()),
        "standard_normal_std": float(z.std(ddof=0)),
        "validation": checks,
    }
    (args.output.parent / "yearly_osse_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output.parent / "YEARLY_OSSE_OBSERVATION_SUMMARY.md").write_text(
        
        "\n".join(
            [
                (
                    "# Formal real-network-driven idealized OSSE observations "
                    f"({str(output.cycle_time.values[0])[:10]} to {str(output.cycle_time.values[-1])[:10]})"
                ),
                "",
                f"- Network: `{args.network}`",
                f"- ERA5 nature run: `{args.era5}`",
                f"- Output: `{args.output}`",
                f"- Cycles: {output.sizes['cycle']}",
                f"- Observations: {output.sizes['obs']:,}",
                f"- Seed: {args.seed}",
                f"- Standard-normal noise: mean={z.mean():.6f}, std={z.std(ddof=0):.6f}",
                "- Each target cycle samples ERA5 at that same cycle time; only January network geometry is repeated.",
                "- Assimilate `synthetic_obs` only; `truth_at_obs` is verification truth.",
                "",
                pd.DataFrame(checks).to_markdown(index=False),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

