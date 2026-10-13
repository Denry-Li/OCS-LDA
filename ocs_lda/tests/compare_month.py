"""Compare a cycling rerun with one archived monthly experiment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--new-method", default="OCS-LDA")
    parser.add_argument("--reference-method", default="PhySP_DA")
    parser.add_argument("--prefix-cycles", type=int, default=0)
    args = parser.parse_args()

    new = pd.read_csv(args.new / "metrics_by_cycle_variable.csv")
    ref = pd.read_csv(args.reference / "metrics_by_cycle_variable.csv")
    if args.prefix_cycles:
        ref = ref[ref["time"].isin(sorted(ref["time"].unique())[:args.prefix_cycles])]
    key = ["time", "variable"]
    new = new.sort_values(key).reset_index(drop=True)
    ref = ref.sort_values(key).reset_index(drop=True)
    if not new[key].equals(ref[key]):
        raise AssertionError("Cycle timestamps or variable labels differ")
    numeric = sorted(set(new.select_dtypes("number")) & set(ref.select_dtypes("number")))
    metric_diff = {}
    for column in numeric:
        a = new[column].to_numpy()
        b = ref[column].to_numpy()
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            raise AssertionError(f"Metric NaN masks differ: {column}")
        finite = np.isfinite(a) & np.isfinite(b)
        if not np.array_equal(np.isinf(a), np.isinf(b)):
            raise AssertionError(f"Metric infinity masks differ: {column}")
        metric_diff[column] = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0

    state_diff = {}
    with xr.open_dataset(args.new / args.new_method / "states.nc") as new_states:
        with xr.open_dataset(args.reference / args.reference_method / "states.nc") as ref_states:
            if args.prefix_cycles:
                ref_states = ref_states.isel(time=slice(0, args.prefix_cycles))
            if not np.array_equal(new_states.time.values, ref_states.time.values):
                raise AssertionError("State timestamps differ")
            if not np.array_equal(new_states.variable.values, ref_states.variable.values):
                raise AssertionError("State variable labels differ")
            for name in ("background", "analysis", "truth", "manifold_background"):
                a = new_states[name].values
                b = ref_states[name].values
                if not np.array_equal(np.isnan(a), np.isnan(b)):
                    raise AssertionError(f"State NaN masks differ: {name}")
                finite = np.isfinite(a) & np.isfinite(b)
                if not np.array_equal(np.isinf(a), np.isinf(b)):
                    raise AssertionError(f"State infinity masks differ: {name}")
                state_diff[name] = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0

    print(json.dumps({
        "cycles": len(new) // 5,
        "metric_max_abs_difference": metric_diff,
        "state_max_abs_difference": state_diff,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
