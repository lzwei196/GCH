# Study results and evidence

The archived study evaluated whether agents could prepare executable calibration experiments, whether those experiments produced useful scientific results, and whether retained formulations supported consistent numerical execution. These results used the study's frozen toolkit and environments. They are historical evidence for the approach, not a new evaluation of the current repository engine.

The companion package is `GRL_SUBMISSION_PACKAGE_2026-09-24`. In the locators below, **`E/` means `04_research_archive/record/experiments/`** and **`V/` means `04_research_archive/record/environments/`**, relative to that package. The package is external to this repository. [TEST_INDEX.json](../paper/metadata/TEST_INDEX.json) and [PAPER_EVIDENCE_MAP.json](../paper/metadata/PAPER_EVIDENCE_MAP.json) map each test ID to its retained evidence, protocol and dependencies.

## Adaptive generality

Six model families received initial single-objective formulations followed by multivariable applications. All six initial specifications executed unchanged. WOFOST, VIC and MODFLOW 6 completed calibration and passed the declared validation gate; HBV, SUMMA and CRHM stopped before optimization because their defaults already met the target. These are six executable formulations, including three successful pre-check stops.

All six multivariable searches completed. Joint acceptance required the component's declared criteria, including comparison with defaults. **Sixteen of 23 variable–metric components passed**, with MODFLOW 6 passing every component.

| Family / archive case | Multivariable targets | Accepted components | Exact test IDs: initial; multivariable |
|---|---|---:|---|
| WOFOST / `01_WOFOST` | Grain mass, biomass | 0/2 | `SINGLE_WOFOST`; `E1_WOFOST_MULTIVARIABLE` |
| HBV / `02_HBV` | Discharge, snow water equivalent | 4/5 | `SINGLE_HBV`; `E1_HBV_MULTIVARIABLE` |
| SUMMA / `03_SUMMA` | Discharge, snow water equivalent | 3/4 | `SINGLE_SUMMA`; `E1_SUMMA_MULTIVARIABLE` |
| VIC / `04_VIC` | Discharge, evapotranspiration | 2/4 | `SINGLE_VIC`; `E1_VIC_MULTIVARIABLE` |
| MODFLOW 6 / `05_MODFLOW6` | Streamflow and filtered-baseflow duration curves | 4/4 | `SINGLE_MODFLOW6`; `E1_MODFLOW6_MULTIVARIABLE` |
| CRHM / `06_CRHM` | Discharge, snow water equivalent | 3/4 | `SINGLE_CRHM`; `E1_CRHM_MULTIVARIABLE` |

Multivariable reports are at `E/1_six_model/six_model_families/<case>/rerun_conv_report.json`; component decisions are in `holdout.per_objective`. Initial results survive as contemporaneous console records at `E/1_six_model/single_objective_phase/CONSOLE_RECORD.md`; their original numerical outputs are unavailable, while the original six contracts are retained.

Validation informed member selection in the multivariable applications. They assess adaptation and selected compromises, rather than independently held-out performance of a calibration-only selection rule. MODFLOW 6's two targets derive from the same discharge record; groundwater-head observations were unavailable.

The component decisions expose meaningful limitations. WOFOST biomass validation NRMSE was 25.8%, below its 30% ceiling but worse than the 21.1% default, so it failed joint acceptance. VIC retained acceptable discharge while evapotranspiration validation NSE fell from 0.05 to −0.29. Completion therefore does not imply adequate representation of every observed quantity.

## Scientific competence

### Knowledge and formulation checks

FSM2 compared five independent generation sessions per information condition using a common fixed runner. Structured knowledge infrastructure (KI) improved first-pass usability. The information packages also differed in content and volume, and generated objectives and budgets differed, so the experiment does not isolate presentation structure alone.

| Check | Result | Test IDs and locator |
|---|---|---|
| Initial FSM2 execution | KI 5/5; raw documentation 3/5; model name only 0/5. KI validation NSE 0.81–0.94 versus default 0.57. | `E2_FSM2_INITIAL`; `E/2_fsm2_knowledge/FSM2_not_memorization/full/JUDGEMENT.json` |
| Prespecified FSM2 repairs | Target and unit repairs made all 5 raw-document and 3 name-only specifications runnable. Unit conversion improved validation NSE in all 8 eligible pairs. | `E2_FSM2_TARGET_REPAIR`, `E2_FSM2_UNIT_REPAIR`; same judgement file and `full/runs/*_r4target/`, `full/runs/*_r4units/` |
| GR4J parameter-domain check | Mean four-parameter empirical-domain overlap: KI 0.81–1.00; name only 0.39–0.60. Three specifications per condition; nine total including raw documentation. | `E3_GR4J_STATIC`; `E/3_gr4j_ranges/B_ki_vs_memory_gr4j/comparison_current_v3.json` |

The FSM2 repairs identify unsupported targets and hours-versus-seconds timescale errors as consequential formulation failures. Repair searches reuse the original generations. The GR4J overlap is a static check in canonical transformed coordinates, not a calibration-skill result.

### Reference checks

| Reference comparison | Generated workflow | Reference | Test ID |
|---|---|---|---|
| GR4J, 1990–1999 calibration window | NSE 0.798785; 1,000 evaluations | airGR NSE 0.798822; 234 evaluations | `E7_E1_GR4J_AIRGR` |
| SAC-SMA, Blanco validation, 3,652 days | NSE 0.6992; KGE 0.6883; PBIAS +24.82% | CAMELS NSE 0.6262; KGE 0.6960; PBIAS −0.53% | `E7_E2_SACSMA_CAMELS` |

Exact comparison values are in `02_source_data/Supporting_Figures/Figure_S5/E1_E2_metrics.json`. Under `E/7_reference_comparisons/E1_E2_reproductions/`, the original generated results are `E1_airGR/calib_report.json` and `E2_SACSMA_CAMELS/results/e2_blanco_result.json`.

These are single reference comparisons. GR4J reports optimization-window fit. The SAC-SMA workflow used 2,702 evaluations under a nominal 3,000 budget and achieved higher NSE with substantially greater volume bias. Its standalone model and Hamon potential evapotranspiration differ from the coupled Snow-17/SAC-SMA/routing reference. Case, periods and reference-method information were supplied during authoring; agreement with published choices is descriptive.

The SAC-SMA parameter check also exposed a supplied-domain limitation: `uzfwm` bounds of 5–150 mm excluded the separate Daymet reference's 490.038–698.212 mm range. None of the eleven fitted values lay within its ten-seed reference envelope. Different forcing and model configurations limit interpretation (`E7_N_PARAMETER_REFERENCE`; `E/7_reference_comparisons/N_expert_match/comparison_tables.json`).

### Matched fixed search versus continuing agent proposals

Ten VIC Tangnaihai paired blocks shared six parameters, an NSE objective and five starting vectors. Each arm completed exactly 150 calibration evaluations, then evaluated its calibration-selected vector once on validation.

| Validation result | Value |
|---|---:|
| Fixed DDS median NSE | 0.872413 |
| Continuing-agent median NSE | 0.875487 |
| Median paired difference, fixed minus agent | −0.002993 |
| 95% complete-block bootstrap interval | [−0.007354, 0.002339] |
| Prespecified non-inferiority margin | −0.02 |

The interval met the margin, with zero optimization-stage language-model calls for fixed DDS. This supports independent numerical search for this formulation and budget. The compact proposer used MADR diagnostics and frozen search guidance; it did **not** reproduce the complete MADR workflow. Ten interrupted agent attempts were excluded under recorded infrastructure-failure rules; all twenty included endpoints were finite without imputation.

Evidence: `E4_VIC_PAIRED`; `E/4_vic_paired_search/A_author_once_vs_in_loop/ANALYSIS_paired_blocks.json`, with `DESIGN_v2.md` and `CAMPAIGN_MANIFEST.json` beside it.

## Consistency and reproducibility

Five GR4J formulations were searched with three seeds each. Six matched KGE-to-NSE interventions produced 21 searches overall. Among the **15 common-NSE endpoints**, the total NSE range was **9.22 × 10⁻⁴** and the largest within-formulation seed range was **2.29 × 10⁻⁴**. Changing KGE to NSE shifted mean NSE by approximately 0.06. Consistency was conditional on the objective and measured over the full optimization window, not a separate validation period.

Evidence: `E5_GR4J_FORMULATIONS`, `E5_GR4J_OBJECTIVE_INTERVENTIONS`; `E/5_gr4j_repeated/C_stability_gr4j/results.json`, particularly `panels.all_nse`.

Eleven archived replay packages each evaluated **one specified parameter vector**: the six model-family cases, paired VIC, repeated GR4J, two reference comparisons and FSM2. All reproduced their comparison metrics; repeated GR4J, reference GR4J and FSM2 metric files were byte-identical. SUMMA required the recorded floating-point vector, and SAC-SMA required the native parameter-file representation. These are individual numerical evaluations, often defaults, rather than complete search or fresh authoring reruns. See `V/<case>/RERUN.md`, `recorded/` and `rerun_evidence/`; for example, test `REPLAY_C_GR4J_L0123001`.

## Inspecting the evidence

These commands list retained inputs and evidence status without running models:

```bash
python paper/reproduce.py list --test E1_VIC_MULTIVARIABLE
python paper/reproduce.py list --test E2_FSM2_INITIAL
python paper/reproduce.py list --test E4_VIC_PAIRED
python paper/reproduce.py list --test E7_E2_SACSMA_CAMELS
python paper/reproduce.py list --test REPLAY_C_GR4J_L0123001
```

The [paper reproduction guide](../paper/README.md) distinguishes saved-result checks, environment checks and numerical reruns. File integrity alone does not establish execution readiness: the exact sealed GR4J Python runtime remains unrecovered. Nine additional cross-provider authored contracts are unavailable; their console records do not constitute a presently rerunnable contract set.
