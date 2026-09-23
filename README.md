# OCS-LDA: core code candidate

**Private, unpublished candidate. No licence or public release has been applied.**

This compact repository contains only the final OCS-LDA model and training
entry point, the frozen six-hour forecast inference path, the point-observation
operator, the hybrid latent-background-covariance cycling path, one unified
evaluation entry point, and the final experiment configurations. It is not a
copy of the full research workspace.

## Included code

| Purpose | Files |
|---|---|
| Final autoencoder and training | `models/ocs_lda.py`, `models/ocs_lda_blocks.py`, `train_bohai.py`, `dataset_bohai.py`, `loss_bohai.py` |
| Six-hour forecast inference | `models/BohaiForecastNet.py`, `models/SwimIRv2.py`, `forecast_inference.py`, `forecast_time_features.py` |
| Point observations | `utils/point_observation_operator.py` |
| Full-year synthetic OSSE observations | `utils/generate_gdas_osse_observations_yearly.py` |
| Hybrid latent 3D-Var | `run_ocs_lda.py`, `cycling_3dvar_bohai.py`, `latent_3dvar_bohai.py` |
| Uniform evaluation | `evaluate_cycling.py` |

The historical configuration key `PhySP_DA` and model class `PhySP_AEv4`
are retained for compatibility with the formal checkpoint. They denote the
manuscript's OCS-LDA method. The two model modules were mechanically assembled
from the archived V3, V4 and shared-block sources; the original project source
was not modified.

`configs/final/` contains the final 2023 Full-observation, No-SP Selective,
No-SP Full-update, No-D2M and Shared-off OCS-LDA configurations. The training
and six-hour forecast model configurations are retained under `configs/`.

## External inputs

The repository does not include original ERA5 or GDAS material, synthetic
observations, the prepared GDAS network NetCDF, trained checkpoints, covariance binaries, or annual analysis
states. A complete run requires the matching ERA5 nature-run and initialization
files, OSSE observation NetCDF, normalization statistics, trained OCS-LDA and
forecast weights, and latent `B_z`. The final YAML files preserve the archived
paths; map those input fields to their released locations before running.

The covariance calculation scripts, GDAS network-construction scripts, historical
architecture variants, LDA and Traditional 3D-Var baselines, figure scripts,
and archived postprocessing are deliberately excluded from this core candidate.
They remain in the private full-code archive. The included full-year OSSE
generator takes a prepared GDAS network NetCDF and the matching ERA5 nature-run
NetCDF; it does not download GDAS or build the network from raw GDAS files.
This repository therefore supports re-running OCS-LDA analyses with prepared
inputs and regenerating synthetic observations from those two prepared inputs,
not rebuilding every input from raw source data.

## Run and evaluate

Install the dependencies in `environment.yml` and run from the repository root.
After mapping input paths in one of the frozen YAML files:

```bash
python run_ocs_lda.py --config configs/final/fullobs_ocs_lda.yaml --output-dir outputs/fullobs --device cuda:0
python evaluate_cycling.py --run-dir outputs/fullobs --output-dir outputs/fullobs/evaluation --warmup 8 --expected-cycles 1460
```

The evaluator reports annual and monthly physical-space RMSE, MAE and PCC.
For each variable, it first computes the metric at each cycle and then averages
over the evaluated cycles. The reported background-to-analysis RMSE reduction
is calculated from annual mean RMSE values; it is not a comparison with LDA.
The first eight cycles are excluded by default.

To regenerate the full-year synthetic observations from prepared inputs:

```bash
python utils/generate_gdas_osse_observations_yearly.py --network path/to/network.nc --era5 path/to/era5_nature_run.nc --output path/to/osse_observations.nc --seed 42 --audit-dir path/to/osse_audit
```

## Validation and limits

Using the archived server inputs, the two-module model strictly loaded the
original D32 `model_state` checkpoint. A clean one-cycle 2023 Full-observation
hybrid-C run, with the three old model files absent, matched the archived
background, analysis, truth and manifold-background arrays exactly (maximum
absolute difference 0.0). This does not establish full-year or clean-machine
reproduction. The next release gate is a run using only the proposed public
code, data and weights.
