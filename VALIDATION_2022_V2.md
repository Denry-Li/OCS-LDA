# OCS-LDA V2: 2022 four-month compatibility validation

This check asks whether the refactored `ocs_lda.OCSLDA` model, using the **original trained D32 checkpoint**, original six-hour forecast checkpoint, original hybrid-C latent `B_z`, and unchanged 2022 ERA5/OSSE inputs, reproduces the archived D32 hybrid-C cycling analyses. It is a code-compatibility check, not a newly trained model or a new scientific experiment.

## Inputs and protocol

- Remote source archive: `/home/lidunlei/Project/BohaiLDA/experiments/cycling_3DVar/validation_2022_seasonal_fullmonth/latent_dim_ablation_24m12_hybrid_C/formal/D32/`.
- Refactored-code validation checkout: `/home/lidunlei/Project/OCS-LDA-refactor-check/`.
- Months: March, June, September and December 2022, each from day 1 at 00:00 UTC through the last day at 18:00 UTC, cycling every six hours.
- Each month begins with eight spin-up cycles that are excluded from the published evaluation statistics, but **included** in the per-cycle/state-file identity checks below.
- The original checkpoint is loaded strictly into `OCSLDA` without a key-conversion map. The validation reuses the archived observation realization, initialization/nature run, forecast model, `B_z`, optimizer settings and random seed.

## Full-month result

| Month | Raw cycles | Evaluated cycles | Max absolute difference in per-cycle numeric metrics | Max absolute difference in saved state arrays |
|---|---:|---:|---:|---:|
| 2022-03 | 124 | 116 | 0.0 | 0.0 |
| 2022-06 | 120 | 112 | 0.0 | 0.0 |
| 2022-09 | 120 | 112 | 0.0 | 0.0 |
| 2022-12 | 124 | 116 | 0.0 | 0.0 |
| **Total** | **488** | **456** | **0.0** | **0.0** |

The comparison matches each cycle timestamp and variable label before comparing values; it checks NaN/infinity masks as well as finite values. The numeric columns compared are analysis/background RMSE, MAE, correlation and bias; observation-space RMSE/chi-square; observation count; and RMSE-reduction percentage. The saved `states.nc` fields compared are `analysis`, `background`, `truth` and `manifold_background`. A reported `0.0` is the maximum observed absolute difference, not merely equality after decimal rounding.

Rerun outputs are under `validation_runs/formal_YYYYMM/` in the validation checkout. Reference outputs are under `formal/D32/YYYYMM/` in the source archive above. The executable comparator is `python -m ocs_lda.tests.compare_month`; for this compatibility run, specify `--new-method PhySP_DA` because the archived scenario YAML/output label was intentionally retained while swapping only the model implementation. The new repository's public runtime name is `OCS-LDA`.

## Fully renamed public entry point

The renamed branch was also checked separately in `/home/lidunlei/Project/OCS-LDA-refactor-final-test/` using the new `configs/validation_2022/202203.yaml`, output label `OCS-LDA` and formal entry point `python -m ocs_lda.run_ocs_lda`. The complete March output is `validation_runs/full_named_202203_wrapper/`. Across all 124 March cycles, the maximum absolute difference from the archived run was **0.0** for every common numeric metric and every checked state array. This catches path/key/method-name integration errors beyond the model-only compatibility check. The nine released cycling YAML scenarios pass the import/config consistency check, and a forward/loss/backward training-step smoke test passes.

The hybrid-C Pressure adapter is installed by the formal entry point. Invoking `python -m ocs_lda.cycling_3dvar_bohai` directly with a hybrid-C YAML bypassed that adapter and produced different analyses. The refactor now rejects this invocation before creating an output directory; the formal wrapper remains the supported entry point. A guarded one-cycle wrapper run reproduced the archived metrics and state arrays exactly.

These checks use the authors' complete server-side research inputs. They do not imply that the compact GitHub/Zenodo release alone includes all ERA5 and prepared-network assets needed for a clean-machine rerun, or that an entirely new model training run has been performed.

## Import and final-configuration checks

The assembled repository passed `python -m compileall -q ocs_lda models utils` and the following tests in the authors' DA environment:

- `python -m ocs_lda.tests.test_imports`: all 17 supported package/runtime modules import, including the public `ocs_lda.OCSLDA`, forecast inference, training, observation operator, cycling wrapper and evaluation.
- `python -m ocs_lda.tests.test_configs`: all nine released cycling YAML configurations (five final 2023 scenarios and four 2022 validation months) resolve to the included D32 model YAML and use the `OCS-LDA` method name.
- `python -m ocs_lda.tests.test_training_step`: model forward pass, structured loss and backward pass produce a finite loss and finite gradients.
- `python -m ocs_lda.tests.test_final_smokes --runs-dir validation_runs`: each of the five final 2023 YAML scenarios completes one cycle through `python -m ocs_lda.run_ocs_lda`, loads the model and hybrid-C `B_z`, and writes finite five-variable analysis states.

| Final scenario | Active branches | Optimized latent elements |
|---|---|---:|
| Full observation | Shared + Wind + Thermo-moist + Pressure | 3200 |
| No-SP Selective | Shared + Wind + Thermo-moist | 2800 |
| No-SP Full-update | Shared + Wind + Thermo-moist + Pressure | 3200 |
| No-D2M | Shared + Wind + Thermo-moist + Pressure | 3200 |
| Shared-off | Wind + Thermo-moist + Pressure | 2000 |

These are entry-point/import smoke tests for the 2023 scenarios, **not** new complete 2023 annual reruns. The separate full-month and four-month numerical-equivalence checks are reported above.
