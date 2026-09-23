"""Summarize the archived per-cycle OCS-LDA physical-space metrics.

RMSE, MAE and PCC are averaged across evaluation cycles for each variable.
The annual RMSE reduction is calculated from those annual mean RMSE values,
not by averaging the per-cycle reduction percentages.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "method",
    "time",
    "variable",
    "background_rmse",
    "analysis_rmse",
    "background_mae",
    "analysis_mae",
    "background_corr",
    "analysis_corr",
}


def summarize(run_dir: Path, output_dir: Path, warmup: int, expected_cycles: int | None):
    source = run_dir / "metrics_by_cycle_variable.csv"
    frame = pd.read_csv(source)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing metric columns in {source}: {missing}")
    frame = frame.loc[frame["method"] == "PhySP_DA"].copy()
    if frame.empty:
        raise ValueError(f"No PhySP_DA rows in {source}")
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    if frame.duplicated(["variable", "time"]).any():
        raise ValueError("Duplicate variable/time rows in per-cycle metrics")
    counts = frame.groupby("variable")["time"].nunique()
    if counts.nunique() != 1:
        raise ValueError(f"Unequal cycle counts by variable: {counts.to_dict()}")
    raw_cycles = int(counts.iloc[0])
    if expected_cycles is not None and raw_cycles != expected_cycles:
        raise ValueError(f"Expected {expected_cycles} cycles, found {raw_cycles}")
    if not 0 <= warmup < raw_cycles:
        raise ValueError(f"warmup must be in [0, {raw_cycles})")

    frame = frame.sort_values(["variable", "time"])
    frame["cycle_index"] = frame.groupby("variable").cumcount()
    evaluated = frame.loc[frame["cycle_index"] >= warmup].copy()
    yearly = (
        evaluated.groupby("variable", as_index=False)
        .agg(
            evaluated_cycles=("time", "size"),
            background_rmse=("background_rmse", "mean"),
            analysis_rmse=("analysis_rmse", "mean"),
            background_mae=("background_mae", "mean"),
            analysis_mae=("analysis_mae", "mean"),
            background_pcc=("background_corr", "mean"),
            analysis_pcc=("analysis_corr", "mean"),
        )
    )
    yearly["background_to_analysis_rmse_reduction_percent"] = (
        1.0 - yearly["analysis_rmse"] / yearly["background_rmse"]
    ) * 100.0
    monthly = (
        evaluated.assign(month=evaluated["time"].dt.month)
        .groupby(["variable", "month"], as_index=False)
        .agg(
            evaluated_cycles=("time", "size"),
            analysis_rmse=("analysis_rmse", "mean"),
            analysis_mae=("analysis_mae", "mean"),
            analysis_pcc=("analysis_corr", "mean"),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(output_dir / "annual_metrics.csv", index=False)
    monthly.to_csv(output_dir / "monthly_metrics.csv", index=False)
    print(f"Evaluated {raw_cycles - warmup} of {raw_cycles} cycles per variable")
    print(yearly.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--expected-cycles", type=int, default=None)
    args = parser.parse_args()
    summarize(args.run_dir, args.output_dir, args.warmup, args.expected_cycles)


if __name__ == "__main__":
    main()
