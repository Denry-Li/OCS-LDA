"""Smoke-test the public package and every supported runtime module."""

import importlib


MODULES = (
    "ocs_lda",
    "ocs_lda.model",
    "ocs_lda.structured_autoencoder",
    "ocs_lda.swin_components",
    "ocs_lda.dataset_bohai",
    "ocs_lda.loss_bohai",
    "ocs_lda.train_bohai",
    "ocs_lda.forecast_inference",
    "ocs_lda.forecast_time_features",
    "ocs_lda.latent_3dvar_bohai",
    "ocs_lda.cycling_3dvar_bohai",
    "ocs_lda.run_ocs_lda",
    "ocs_lda.evaluate_cycling",
    "models.BohaiForecastNet",
    "models.SwimIRv2",
    "utils.point_observation_operator",
    "utils.generate_gdas_osse_observations_yearly",
)


def main() -> None:
    for name in MODULES:
        importlib.import_module(name)
    from ocs_lda import OCSLDA

    assert OCSLDA.__module__ == "ocs_lda.model"
    print(f"imports passed: {len(MODULES)} modules; public model={OCSLDA.__name__}")


if __name__ == "__main__":
    main()
