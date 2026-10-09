# OCS-LDA

This repository contains the core implementation of observation-conditioned structured latent data assimilation (OCS-LDA) for regional multivariate point observations over the Bohai–Yellow Sea. Historical code and result labels containing `PhySP_DA` refer to the same method.

## Contents

- Structured autoencoder: [`models/ocs_lda.py`](models/ocs_lda.py) and [`models/ocs_lda_blocks.py`](models/ocs_lda_blocks.py)
- Autoencoder training and six-hour forecast model: [`ocs_lda/train_bohai.py`](ocs_lda/train_bohai.py) and [`models/BohaiForecastNet.py`](models/BohaiForecastNet.py)
- Point-observation operator and OSSE generation: [`utils/point_observation_operator.py`](utils/point_observation_operator.py) and [`utils/generate_gdas_osse_observations_yearly.py`](utils/generate_gdas_osse_observations_yearly.py)
- Hybrid-`B_z` cycling assimilation and evaluation: [`ocs_lda/run_ocs_lda.py`](ocs_lda/run_ocs_lda.py) and [`ocs_lda/evaluate_cycling.py`](ocs_lda/evaluate_cycling.py)
- Final experiment configurations: [`configs/final/`](configs/final/)

The Python implementation lives in the `ocs_lda/` package. Run its training,
cycling and evaluation entry points from the repository root with `python -m`;
`environment.yml` stays at the root for environment setup.

## Data and model artifacts

The companion [Zenodo dataset record](https://zenodo.org/records/22915966) contains the author-generated 2022–2023 full-observation OSSE files, a minimum evaluation/source-data package, and these three artifacts for the D32 hybrid-C experiment:

| File | Purpose |
|---|---|
| `D32_best.pt` | Trained OCS-LDA D32 autoencoder checkpoint |
| `ai6h_stage2_ar4_best.pt` | Trained six-hour forecast model checkpoint |
| `ocs_lda_D32_hybrid_C.pt` | Fitted D32 hybrid-C latent background-error covariance (`B_z`) |

The record is currently an unpublished draft; the link will become publicly accessible when the authors publish the deposit. The permanent dataset DOI will be added after registration. See the [reproducibility guide](REPRODUCIBILITY.md) for the environment, required external inputs, path mapping, commands, and result checks. In particular, the full ERA5 regional nature-run and initialization states are **not** bundled with these five Zenodo files.

Original ERA5 and GDAS/PREPBUFR records are not redistributed in this repository. Obtain them from the [Copernicus Climate Data Store](https://doi.org/10.24381/cds.adbb2d47) and [NOAA NCEI](https://www.ncei.noaa.gov/products/weather-climate-models/global-data-assimilation), respectively.

## Paper

The OCS-LDA manuscript is under preparation. The paper citation and link will be added after publication.

## Licence

The authors' licensable code contributions are under [Apache-2.0](LICENSE). The author-generated Zenodo dataset is under CC BY 4.0; the original ERA5 and GDAS/PREPBUFR products retain their own terms. Adapted Swin-family components and applicable notices are identified in [Third-party notices](THIRD_PARTY_NOTICES.md). The external LDA comparator is not bundled.
