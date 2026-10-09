# OCS-LDA: code and reproducibility guide

**Release candidate.** Source is provided under Apache-2.0 for the authors' licensable contributions, with upstream SwinIR/Swin Transformer/KAIR conditions documented in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The GitHub repository remains private until the authors publish it; no public software DOI is claimed here.

The formal OCS-LDA experiments have been rerun in the authors' research
environment. This compact repository documents the inputs, commands and checks
needed to repeat the analysis with the released core implementation. A rerun
using the authors' complete assets does not imply that every large input is
bundled in Git.

## Included code

| Purpose | Files |
|---|---|
| Final autoencoder and training | `models/ocs_lda.py`, `models/ocs_lda_blocks.py`, `ocs_lda/train_bohai.py`, `ocs_lda/dataset_bohai.py`, `ocs_lda/loss_bohai.py` |
| Six-hour forecast inference | `models/BohaiForecastNet.py`, `models/SwimIRv2.py`, `ocs_lda/forecast_inference.py`, `ocs_lda/forecast_time_features.py` |
| Point observations | `utils/point_observation_operator.py` |
| Full-year synthetic OSSE observations | `utils/generate_gdas_osse_observations_yearly.py` |
| Hybrid latent 3D-Var | `ocs_lda/run_ocs_lda.py`, `ocs_lda/cycling_3dvar_bohai.py`, `ocs_lda/latent_3dvar_bohai.py` |
| Uniform evaluation | `ocs_lda/evaluate_cycling.py` |

The historical configuration key `PhySP_DA` and model class `PhySP_AEv4`
are retained for compatibility with the formal checkpoint. They denote the
manuscript's OCS-LDA method. The two model modules were mechanically assembled
from the archived V3, V4 and shared-block sources; the original project source
was not modified.

`configs/final/` contains the final 2023 Full-observation, No-SP Selective,
No-SP Full-update, No-D2M and Shared-off OCS-LDA configurations. The training
and six-hour forecast model configurations are retained under `configs/`.

## External inputs

The Git repository does not include original ERA5 or GDAS material, synthetic
observations, the prepared GDAS network NetCDF, trained checkpoints, fitted
covariance binaries, or annual analysis states. The companion Zenodo dataset
includes the 2022–2023 full-observation OSSE files, minimum evaluation/source
data, OCS-LDA D32 and six-hour forecast checkpoints, and fitted D32 hybrid-C
`B_z`. A cycling run additionally requires the
matching 40 × 40 ERA5 nature-run and two initialization states, OSSE
observation NetCDF, normalization statistics, trained OCS-LDA and 6-h forecast
weights, and fitted latent `B_z`. The final YAML files preserve the archived
paths as provenance. **Copy a YAML before changing paths; do not present a
locally remapped copy as the exact executed configuration.**

| Field in `configs/final/fullobs_ocs_lda.yaml` | Input to supply |
|---|---|
| `forecast_checkpoint` | Matching trained forecast weights |
| `physp_config`, `physp_checkpoint` | Included D32 architecture YAML and matching OCS-LDA weights |
| `physp_bz` | Fitted D32 semantic-branch covariance with Pressure spatial variance for hybrid C |
| `norm_stats` | Included `data/stats/norm_stats_train.json` |
| `initialization_file` | ERA5 states at the two configured 2022-12-31 times |
| `nature_file` | 2023 ERA5 reference states on the 40 × 40 grid |
| `observation_file` | 2023 OSSE NetCDF with the formal grid, cycle times and observation realization |

The common YAML also contains `lda_*` and `traditional_*` paths. When
`methods` contains only `PhySP_DA`, those comparator artifacts are **not**
loaded; they are not additional requirements for an OCS-LDA-only run. The
`PhySP_DA` method label denotes OCS-LDA in the manuscript.

The covariance calculation scripts, GDAS network-construction scripts, historical
architecture variants, LDA and Traditional 3D-Var baselines, figure scripts,
and archived postprocessing are deliberately excluded from this core candidate.
They remain in the private full-code archive. The included full-year OSSE
generator takes a prepared GDAS network NetCDF and the matching ERA5 nature-run
NetCDF; it does not download GDAS or build the network from raw GDAS files.
This repository therefore supports re-running OCS-LDA analyses with prepared
inputs and regenerating synthetic observations from those two prepared inputs,
not rebuilding every input from raw source data.

The Zenodo `observations.zip` contains full-observation OSSE files for 2022 and
2023. Before using its 2023 file in place of the historical path, check the
NetCDF variables, 40 × 40 target grid, cycle times and observation realization
against the formal run manifest. The No-SP and No-D2M YAML files refer to
historical scenario-specific files. The observation loader supports selecting
variables from a full-observation file, but an exact rerun requires checking
that the filtered records equal the formal scenario inputs.

Original ERA5 can be obtained from the [Copernicus Climate Data Store](https://doi.org/10.24381/cds.adbb2d47)
and GDAS/PREPBUFR from [NOAA NCEI](https://www.ncei.noaa.gov/products/weather-climate-models/global-data-assimilation).
This compact repository does not provide the complete download and
preprocessing chain from those upstream products to the formal 40 × 40 inputs.

The author-generated 2022–2023 synthetic OSSE observations, minimum
figure/evaluation dataset, and three D32 model artifacts are in the companion
[Zenodo Dataset draft](https://zenodo.org/records/22915966). The three artifact
files are `D32_best.pt`, `ai6h_stage2_ar4_best.pt`, and
`ocs_lda_D32_hybrid_C.pt`. The record remains unpublished; its final DOI must
be inserted here after publication. The data record uses CC BY 4.0 for the
authors' generated material; it does not
relicense original ERA5 or GDAS/PREPBUFR data.

## Environment, run and evaluate

`environment.yml` records the intended Python 3.10.20 scientific environment.
Create it with a Conda-compatible manager; GPU execution also requires a
PyTorch build compatible with the local CUDA driver. Clean-machine installation
of this compact release has not been independently verified.

```bash
conda env create -f environment.yml
conda activate ocs-lda-reproduction
```

To retrain the autoencoder or forecast model, first prepare the 2016–2021
training samples and 2022 validation samples expected by the respective
configuration's `samples_root`. Work on local YAML copies with mapped paths:

```bash
python -m ocs_lda.train_bohai --config configs/local/physp_v4_d32_mse_seed42.yaml --device cuda:0
python -m ocs_lda.train_bohai --config configs/local/bohai_forecast_net_era5_5var_40x40_nodown_time2_ar4.yaml --device cuda:0
```

The original training YAML files are under
`configs/Hyperparameter Optimization/search_latent_dim/` and `configs/3DVar/`.
Retraining requires their prepared training data. The included code does not
regenerate the formal NMC `B_z`; use the matching archived covariance artifact for the
reported cycling configuration.

Run from the repository root after mapping the input paths in a **copy** of a
final YAML. The following `configs/local/` path is an example local copy, not
a file supplied by this repository:

```bash
python -m ocs_lda.run_ocs_lda --config configs/local/fullobs_ocs_lda.yaml --output-dir outputs/fullobs --device cuda:0
python -m ocs_lda.evaluate_cycling --run-dir outputs/fullobs --output-dir outputs/fullobs/evaluation --warmup 8 --expected-cycles 1460
```

The runner refuses to overwrite a non-empty output directory by default.
The full 2023 protocol has 1460 six-hour cycles; the first eight are spin-up,
leaving **1452 evaluated cycles**. The cycling run writes `manifest.json`,
`metrics_by_cycle_variable.csv`, `metrics_summary.csv` and method-specific
`states.nc`/optimizer diagnostics. The evaluator writes annual and monthly
physical-space RMSE, MAE and PCC to `annual_metrics.csv` and
`monthly_metrics.csv`.
For each variable, it first computes the metric at each cycle and then averages
over the evaluated cycles. The reported background-to-analysis RMSE reduction
is calculated from annual mean RMSE values; it is not a comparison with LDA.

For No-SP, No-D2M or Shared-off, copy the corresponding YAML under
`configs/final/`, map its inputs, and use a separate output directory. No-SP
Selective excludes Pressure Private; No-SP Full-update retains it. No-D2M
retains Thermo-moist Private because T2M observations remain available.

To regenerate the full-year synthetic observations from prepared inputs:

```bash
python utils/generate_gdas_osse_observations_yearly.py --network path/to/network.nc --era5 path/to/era5_nature_run.nc --output path/to/osse_observations.nc --seed 42 --audit-dir path/to/osse_audit
```

## How to verify a rerun

1. Record the Git commit, local YAML, environment, and SHA-256 hashes of the
   forecast/OCS-LDA checkpoints, `B_z`, ERA5 fields and observation file.
   `manifest.json` records key configuration and checkpoint hashes.
2. Confirm that all 1460 cycles complete without skipped/failed cycles and
   that each of the five variables has 1452 evaluated cycles.
3. Compare annual/monthly RMSE, MAE and PCC with the per-cycle evaluation and
   source-data CSVs in the separate Zenodo dataset. Metrics are computed per
   cycle on the 40 × 40 grid and then averaged over evaluated cycles.
4. For exact formal-run comparison, check observation and model-artifact hashes
   and selected `states.nc` fields. If hardware, software or inputs differ,
   report them and use an explicit numerical tolerance rather than claiming
   bitwise identity.

The authors have rerun the experimental workflow in their research environment.
The specific compatibility check documented for this compact code assembly is:

Using the archived server inputs, the two-module model strictly loaded the
original D32 `model_state` checkpoint. A clean one-cycle 2023 Full-observation
hybrid-C run, with the three old model files absent, matched the archived
background, analysis, truth and manifold-background arrays exactly (maximum
absolute difference 0.0). This check does not by itself establish a full-year
*independent-machine* run using only the compact public code/data package.
That is distinct from the authors' successful rerun with complete research
assets. The compact Git/Zenodo package now includes the D32 weights and fitted
hybrid-C `B_z`, but not the full ERA5 regional state archives and all prepared
inputs needed for a fresh cycling run.

## Licence and attribution

The authors' licensable contributions are provided under Apache-2.0 (`LICENSE`).
Adapted SwinIR-family portions retain their upstream attribution and
applicable Apache-2.0/MIT notices in `THIRD_PARTY_NOTICES.md` and
`THIRD_PARTY_LICENSES/`. The external LDA comparator is not redistributed or
relicensed here. A permanent software DOI is a separate release step; none
is claimed in this draft.
