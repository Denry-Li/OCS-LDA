"""Validate one-cycle outputs for each released 2023 scenario."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


EXPECTED = {
    "fullobs_ocs_lda": ("shared+wind+thermo_moist+pressure", 3200),
    "no_sp_selective": ("shared+wind+thermo_moist", 2800),
    "no_sp_full_update": ("shared+wind+thermo_moist+pressure", 3200),
    "no_d2m_ocs_lda": ("shared+wind+thermo_moist+pressure", 3200),
    "shared_off_ocs_lda": ("wind+thermo_moist+pressure", 2000),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    args = parser.parse_args()
    for name, (branches, optimized_elements) in EXPECTED.items():
        run = args.runs_dir / f"final_smoke_{name}"
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["methods"] == ["OCS-LDA"], name
        assert manifest["ocs_lda_model_arch"] == "ocs_lda", name
        assert manifest["ocs_lda_checkpoint_sha256"], name
        assert manifest["ocs_lda_bz_sha256"], name
        optimizer = pd.read_csv(run / "OCS-LDA/optimizer_by_cycle.csv")
        assert len(optimizer) == 1, name
        row = optimizer.iloc[0]
        assert row["active_branches"] == branches, name
        assert int(row["latent_full_numel"]) == 3200, name
        assert int(row["latent_optimized_numel"]) == optimized_elements, name
        assert int(row["point_obs_count"]) > 0, name
        with xr.open_dataset(run / "OCS-LDA/states.nc") as states:
            assert states.sizes["time"] == 1, name
            assert states.sizes["variable"] == 5, name
            assert np.isfinite(states["analysis"].values).all(), name
        print(f"{name}: {branches} ({optimized_elements}/3200), one cycle passed")


if __name__ == "__main__":
    main()
