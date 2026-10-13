"""Check the released OCS-LDA configurations and local model imports."""

from pathlib import Path

import yaml

from ocs_lda import OCSLDA


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    model_path = root / "configs/final/ocs_lda_d32_model.yaml"
    model_cfg = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    assert model_cfg["model_arch"] == "ocs_lda"
    assert OCSLDA(model_cfg).latent_dim == 32

    files = sorted((root / "configs/final").glob("*.yaml"))
    files += sorted((root / "configs/validation_2022").glob("*.yaml"))
    scenarios = 0
    for path in files:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if "methods" not in config:
            continue
        assert config["methods"] == ["OCS-LDA"], path
        assert root / config["ocs_lda_config"] == model_path, path
        assert config["ocs_lda_checkpoint"], path
        assert config["ocs_lda_bz"], path
        assert config["experiment_pressure_parameterization"] == "spatial_diagonal", path
        scenarios += 1
    assert scenarios == 9, scenarios
    print(f"config/import checks passed: {scenarios} cycling scenarios")


if __name__ == "__main__":
    main()
