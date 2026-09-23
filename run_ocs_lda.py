"""Run the final OCS-LDA hybrid-C cycling analysis.

Use this script from the repository root with a frozen config under
configs/final. The implementation preserves the formal experiment's
Pressure-spatial-diagonal adapter without its original server/GPU paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cycling_3dvar_bohai as cycling
import latent_3dvar_bohai as latent

_ORIGINAL_SELECT = cycling.select_latent_bz_mode
_ORIGINAL_CHANNEL_LOSS = latent.channel_background_loss
_ORIGINAL_SLICES = cycling.branch_slices_from_model


def _mixed_channel_loss(z, zb, item, reduction="mean", eps=1e-8):
    if "hybrid_spatial_variance" not in item:
        return _ORIGINAL_CHANNEL_LOSS(z, zb, item, reduction=reduction, eps=eps)
    variance = item["hybrid_spatial_variance"].to(z)
    total = (0.5 * (z - zb).square() / (variance.unsqueeze(0) + eps)).sum()
    if reduction == "sum":
        return total
    if reduction == "mean":
        return total / (z.shape[0] * z.shape[2] * z.shape[3])
    raise ValueError(reduction)


def install_hybrid_c(config):
    mode = config.get("experiment_pressure_parameterization", "block")
    if mode not in {"block", "spatial_diagonal", "channel_diagonal"}:
        raise ValueError(f"Unsupported Pressure parameterization: {mode}")
    if config["physp_bz_scale"] != 1.0:
        raise ValueError("Formal hybrid-C configuration requires physp_bz_scale=1")
    if not all(value == 1.0 for value in config["physp_bz_branch_scales"].values()):
        raise ValueError("Formal hybrid-C branch scales must all equal 1")
    full_slices = {}

    def capture_slices(model):
        slices = _ORIGINAL_SLICES(model)
        if slices is not None:
            full_slices.update(slices)
        return slices

    cycling.branch_slices_from_model = capture_slices

    def select(background, kind, name, branch_names=None):
        selected = _ORIGINAL_SELECT(background, kind, name, branch_names)
        if name != "PhySP_DA" or mode == "block":
            return selected
        if kind != "semantic_branch_block":
            raise ValueError("Hybrid C requires semantic_branch_block B_z")
        if "pressure" not in selected["items"]:
            selected["experiment_pressure_parameterization"] = mode
            return selected
        item = dict(selected["items"]["pressure"])
        if mode == "channel_diagonal":
            cholesky = item["cholesky"]
            diagonal = cholesky.square().sum(dim=1)
            item["cholesky"] = torch.diag(diagonal.sqrt())
            item.pop("cov_loaded", None)
        else:
            diagonal = latent.load_latent_bz(
                config["physp_bz"],
                item["cholesky"].device,
                item_names=["latent"],
                load_covariance=False,
            )["items"]["latent"]["diag_var_spatial"]
            pressure_slice = full_slices["pressure"]
            if pressure_slice.start != 28 or pressure_slice.stop != 32:
                raise ValueError("Unexpected D32 Pressure slice")
            item["hybrid_spatial_variance"] = diagonal[pressure_slice].clone()
            if item["hybrid_spatial_variance"].shape != (4, 10, 10):
                raise ValueError("Unexpected Pressure spatial variance shape")
        selected["items"] = dict(selected["items"], pressure=item)
        selected["experiment_pressure_parameterization"] = mode
        return selected

    cycling.select_latent_bz_mode = select
    latent.channel_background_loss = _mixed_channel_loss


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "Usage: python run_ocs_lda.py --config CONFIG "
            "--output-dir DIR [--device auto|cpu|cuda:0]"
        )
        return
    try:
        config_path = Path(sys.argv[sys.argv.index("--config") + 1])
    except (ValueError, IndexError) as exc:
        raise SystemExit("--config is required") from exc
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    install_hybrid_c(config)
    cycling.main()


if __name__ == "__main__":
    main()
