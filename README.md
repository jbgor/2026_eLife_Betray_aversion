# Analysis code for the choice, decision-time, and EEG results

This folder is the self-contained analysis package for the manuscript. It contains the experimental files used by the scripts, the stored model fits and numerical results, and the Python and R code that produce the figures.


| Folder                      | Contents                                                                                                                                    |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `raw_data/`                 | `single.xlsx` (risk), `double.xlsx` (cooperation), cue-locked P3 and LPP, and `Rating.xlsx`.                                                |
| `codes/`                    | Python code for choice-model estimation, and `analysis.R` for the mixed models and figures. Paths are resolved from each script's location. |
| `results/fits/`             | Per-subject maximum-likelihood fits for Model 0, Model 1, and Model 2.                                                                      |
| `results/model_comparison/` | Mean AIC, BIC, and 10-fold CVIC, subject-level scores, and the Model 2 parameter summary. The CVIC column in the tables is named `looic`.   |
| `results/recovery/`         | Parameter-recovery and model-recovery tables.                                                                                               |
| `results/simulation/`       | Cell means from the 500 Model 2 choice simulations, by instructed probability and by principal.                                             |
| `results/derived/`          | Experimental trials joined with stored Model 2 utilities and cue-locked P3 and LPP.                                                         |
| `results/downstream/`       | Fixed effects, loss-versus-betrayal contrasts, task descriptives, and the affective-rating ANOVA.                                           |
| `figures/main_figures/`     | Choice, decision time, affective ratings, model comparison, Model 2 parameter recovery, and cooperation-task outcome plots.                 |
| `figures/supp_figures/`     | Parameter distributions, the three-model recovery heatmap, model recovery, and the Model 2 simulation figures.                              |


Models are M0 random choice, M1 cumulative prospect theory, and M2 cumulative prospect theory plus a cooperation-specific betrayal cost. 

---



## 1. System requirements



### Operating system

The scripts run on Linux, macOS, and Windows. No special hardware is required to redraw the stored results. Repeating the 500 choice simulations takes about one minute. Repeating parameter recovery (500 simulated participants per model) or model recovery (50 group data sets per model) needs a multi-core machine.

### Software

Python **≥ 3.10**, with:

- `numpy`
- `pandas`
- `scipy`
- `openpyxl`

R, with:

- `lme4`
- `lmerTest`
- `MASS`
- `ggplot2`
- `dplyr`
- `tidyr`
- `patchwork`
- `emmeans`
- `scales`

Mixed models and figures are written in R. `Rscript` must be on the path.

---



## 2. Installation

```bash
python -m pip install numpy pandas scipy openpyxl
```

In R:

```r
install.packages(c("lme4", "lmerTest", "MASS", "ggplot2", "dplyr", "tidyr", "patchwork", "emmeans", "scales"))
```

No path variables need to be edited. Each script locates `raw_data/`, `results/`, and `figures/` from the code file’s location. Work from `codes/`.

---



## 3. Demo

These commands use the stored fits, recovery tables, and simulation tables. `analysis.R` refits the mixed models and draws every figure at 1200 dpi.

```bash
cd codes
PYTHON=python Rscript analysis.R
```

`PYTHON` should point to the interpreter that has `openpyxl`. It is used only to read `Rating.xlsx`.

To rebuild the trial-utility table from the stored Model 2 parameters, without repeating estimation:

```bash
python fitting.py --skip-fit --skip-loo
```

`python simulation.py` repeats the 500 choice simulations and rewrites `results/simulation/`. Run `Rscript analysis.R` afterward to redraw the simulation figures.

**Expected output**

- `figures/main_figures/fig_behavior_choice.png` — acceptance by instructed probability and by principal.
- `figures/main_figures/fig_behavior_rt.png` — decision time by instructed probability and by principal.
- `figures/main_figures/fig_affective_rating.png` — valence ratings after good and bad outcomes in each task.
- `figures/main_figures/fig_model_ic.png` — mean AIC, BIC, and CVIC, and subject-level wins.
- `figures/main_figures/fig_parameter_recovery_heatmap_M2.png` and `fig_parameter_recovery_scatter_M2.png` — Model 2 parameter recovery.
- `figures/main_figures/fig_utility_effects.png` — cooperation-task population curves for choice, decision time, P3, and LPP against scaled loss and betrayal values.
- `figures/supp_figures/fig_m2_params.png` and `fig_kappa.png` — Model 2 parameter distributions.
- `figures/supp_figures/fig_parameter_recovery_heatmap.png` — parameter recovery for Models 1 and 2.
- `figures/supp_figures/fig_parameter_recovery_scatter.png` — parameter-recovery scatters. The first row is M0 and M1 (six panels); the second row is M2. A gray bar above each model's panels gives the model name.
- `figures/supp_figures/fig_model_recovery_confusion.png` — group-level model recovery.
- `figures/supp_figures/fig_simulation.png` — observed and simulated acceptance, with instructed probability above principal. Risk and Cooperation share one legend and one set of column titles.
- `figures/supp_figures/fig_ev_rt.png` — predicted decision time against normalized expected-value difference.
- `results/downstream/fixed_effects.csv`, `loss_vs_defect_contrasts.csv`, and `affective_rating_anova.csv`. The same folder also holds the design-level slopes, expected-value trends, and the joint ERP trend.
- `results/simulation/simulation_by_probability.csv` and `simulation_by_stake.csv`.

On a normal desktop, `analysis.R` takes several minutes because of the generalized linear mixed model. `simulation.py` takes about one minute and does not redraw the figures.

---



## 4. Instructions for use

Paths below are relative to this folder. Set the working directory to `codes/` before running a script.

### 4.1 Layout

```
upload_package/
  raw_data/                     single.xlsx, double.xlsx, Cue2P3400500.mat,
                                Cue2LPP600800.mat, Rating.xlsx
  codes/                        model.py, fitting.py, analysis.R,
                                parameter_recovery.py, model_recovery.py,
                                simulation.py
  results/fits/                 random_model, prospect_model, aversion_linear_model
  results/model_comparison/     AIC, BIC, CVIC, parameter summary
  results/recovery/parameter/   parameter-recovery tables
  results/recovery/model/       model-recovery counts and rates
  results/simulation/           500-simulation cell means
  results/derived/              merged_behavior_erp_utilities.csv
  results/downstream/           lme4 fixed effects and contrasts
  figures/main_figures/
  figures/supp_figures/
```



### 4.2 Estimation

`model.py` defines Models 0–2 and reads `raw_data/single.xlsx` and `raw_data/double.xlsx`. Probability codes 1–5 are 0.1, 0.3, 0.5, 0.7, and 0.9. Principals are 0.2, 0.4, 0.6, 0.8, and 1. `fitting.py` estimates each participant by L-BFGS-B from 20 random starting points, computes the 10-fold CVIC, and writes `results/derived/merged_behavior_erp_utilities.csv` by joining the stored Model 2 parameters to each trial. The files in `results/fits/` are the manuscript fits. Running `python fitting.py` without `--skip-fit` repeats that estimation and overwrites them.

### 4.3 Recovery

Parameter recovery simulates 500 participants per model from uniform draws on the estimation bounds, using the observed trial structure, and refits the same model. Summaries are in `results/recovery/parameter/`. Model recovery simulates 50 group data sets per true model, fits Models 0–2, and selects the model with the lowest summed 10-fold CVIC. Counts are in `results/recovery/model/`. Neither recovery is repeated by the demo commands.

### 4.4 Choice simulation

`simulation.py` keeps each participant’s stored Model 2 parameters fixed and redraws choices from the fitted acceptance probabilities. For each of 500 replicates it draws a Bernoulli choice on every experimental trial and writes the cell means under `results/simulation/`. `analysis.R` draws `figures/supp_figures/fig_simulation.png` from those tables. The shaded band is the central 95% of the group-mean acceptance rates. The black line is the observed mean. The upper row uses instructed probability and the lower row uses principal. Risk and Cooperation are columns and share the legend and column titles. The random seed is 20260610.

### 4.5 Mixed models and figures

`analysis.R` reads `results/derived/merged_behavior_erp_utilities.csv` and `raw_data/Rating.xlsx`, fits the mixed models, and draws the figures. Reading the rating workbook uses Python with `openpyxl`; set `PYTHON` if `python` is not that interpreter. Probability codes 1–5 enter the expected-value calculation as 0.1, 0.3, 0.5, 0.7, and 0.9. Cooperation-trial models are:

- `glmer(Choice ~ zloss + zdefect + (zloss + zdefect | Subject), family = binomial)`
- `lmer(RT ~ zloss + zdefect + (zdefect | Subject), REML = FALSE)`
- `lmer(MonP3 ~ zloss + zdefect + (zloss | Subject), REML = FALSE)`
- `lmer(MonLPP ~ zloss + zdefect + (zloss | Subject), REML = FALSE)`

Supplementary models are the quadratic expected-value models for decision time, P3, and LPP; the loss-by-betrayal interactions for P3 and LPP; the design-level P3 and LPP models; and the joint P3/LPP model. Design-level simple slopes are evaluated at scaled probability and principal of −1 and 1.

P3 is the mean of P1, Pz, and P2 from 400–500 ms (`Cue2P3400500.mat`). LPP is the same electrodes from 600–800 ms (`Cue2LPP600800.mat`). Affective ratings come from the `coop` sheet of `Rating.xlsx`: `s2` and `s0` are good and bad outcomes in the risk task, and `d2` and `d0` are the same outcomes in the cooperation task. Participant 51 is excluded. The repeated-measures ANOVA is `Rating ~ FB * task + Error(Subject / (FB * task))`.

### 4.6 Numerical files used in the manuscript

- `results/model_comparison/model_comparison.csv` — mean AIC, BIC, and CVIC
- `results/model_comparison/subject_ic.csv` — subject-level scores; CVIC wins are 34 / 15 / 0 for Models 2 / 1 / 0
- `results/model_comparison/param_summary_M2.csv` — Model 2 parameter means
- `results/recovery/parameter/parameter_recovery_summary_M2.csv` — Model 2 recovery correlations
- `results/recovery/model/model_recovery_rates.csv` — recovery rates 1.00, 0.80, and 1.00
- `results/simulation/simulation_by_probability.csv` and `simulation_by_stake.csv` — observed means and simulation ranges
- `results/downstream/fixed_effects.csv` and `loss_vs_defect_contrasts.csv` — choice, decision time, P3, and LPP
- `results/downstream/affective_rating_anova.csv` and `affective_rating_means.csv` — valence ratings

