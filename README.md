# accidents-rainfall-br

Code and result tables for the manuscript *Hourly rainfall and crash risk on Brazil's federal highways: a time-stratified case-crossover study linking 323,000 crashes to automatic weather stations, 2017–2025* (submitted to *Accident Analysis & Prevention*, 2026).

All crashes recorded by the Brazilian Federal Highway Police in 2017–2025 (632,659) are linked to hourly precipitation from the nearest INMET automatic station within 25 km (323,501 crashes, 442 stations), and odds ratios are estimated with a time-stratified case-crossover design and conditional logistic regression with station-clustered errors. Crash odds rise with rainfall intensity (OR 1.27 to 1.58), remain elevated for about two hours after rain stops and are higher after dry spells, with the largest effects for run-off-road and head-on crashes on rural segments. Sensitivity analyses for station data loss and a leave-one-year-out calibration check complete the study.

## Layout

| Folder | Content |
|---|---|
| `scripts/` | PRF and INMET processing, case-crossover dataset, conditional logistic regression (`clogit.py`), subgroup and sensitivity analyses, predictive calibration check, figures and the manuscript/supplement builders |
| `dados/` | INMET station inventories per year (small public metadata). Raw PRF and INMET files go here but are not redistributed |
| `resultados/` | Case-crossover estimates (`cc_resultados.csv`), lags, dose–response, subgroups, attributable fraction, agreement PRF × INMET, predictive LOYO results and revision tables (`rev17_*`) |
| `figuras/` | Figs. 1–4 of the manuscript (PNG; the submission uses 300 dpi TIFF) |

## Pipeline

1. `scripts/prf_painel.py` — reads the PRF `datatran` CSV files → `dados/prf_acidentes_2017_2025.parquet`.
2. `scripts/inmet_chuva.py` — hourly precipitation from the INMET automatic-station zips → `dados/inmet_chuva_horaria_<year>.parquet`, `dados/inmet_estacoes_<year>.csv`.
3. `scripts/cc_build.py` — links each crash to the nearest station (≤ 25 km) and builds the time-stratified case-crossover sets → `dados/cc_<year>.parquet`.
4. `scripts/analise_cc.py`, `analise_extra.py` (+ `clogit.py`) — conditional logistic regression by intensity, lag, antecedent dry days, crash type, severity and area → `resultados/cc_resultados.csv`, `defasagens.csv`, `extras.csv`, `fracao_atribuivel.csv`, `concordancia_inmet_prf.csv`.
5. `scripts/preditivo.py` — daily station-level crash counts, GLM and LightGBM with leave-one-year-out validation → `preditivo_*.csv`.
6. `scripts/revisao_17-10.py`, `revisao_17-10_missing_extra.py` — revision analyses: station data loss, dose–response test, distance × land use, annual ORs by class → `rev17_*.csv`.
7. `scripts/figuras.py` — Figs. 1–4; `scripts/build_docx.py`, `build_supp.py`, `build_aux_docx.py` — manuscript, supplement, highlights and cover letter (sources not included).

Run the scripts from any working directory: every path is resolved relative to the script's own location (`os.path.dirname(__file__)`).

## Data

Crash records: Polícia Rodoviária Federal, open data `datatran` 2017–2025 (https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos; open government data, CC BY-like attribution). Hourly precipitation: INMET automatic stations, yearly zips (https://portal.inmet.gov.br/dadoshistoricos; open data). Place the raw files under `dados/prf/` and `dados/inmet/` as named in `scripts/prf_painel.py` and `scripts/inmet_chuva.py`; all derived parquets are rebuilt by the pipeline.

Raw input files are **not** redistributed in this repository; they are large and/or belong to third parties. Download them from the sources above and place them as described, or start from the derived tables in `resultados/`.

## Citation

See `CITATION.cff` (co-authors to be added). Licence: MIT.

## Archive

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23461547.svg)](https://doi.org/10.5281/zenodo.23461547)

Version v1.0.0: https://doi.org/10.5281/zenodo.23461548
