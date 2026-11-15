# Statistical Reliability of Explainable AI Evaluations

Data and code for the paper:

> Okoli, D. C. *Statistical Reliability of Explainable AI Evaluations: A Systematic Review and Experiment on Uncertainty, Reproducibility, and Sensitivity to Evaluation Choices in Feature-Attribution Benchmarks.*

The study has two parts:

1. **Systematic review.** We looked at how published comparisons of feature-attribution methods (SHAP, LIME, Grad-CAM and others) report sample sizes, random seeds, repeated runs, confidence intervals, statistical tests and code.
2. **Experiment.** We measured how the masking baseline, the metric, the random seed and the number of evaluation examples change which explanation method "wins".

## Files

| File | What it contains |
|---|---|
| `screening_records.csv` | The 646 randomly sampled records and their title/abstract screening decisions. Searches were run in Scopus and IEEE Xplore on 14 Nov 2026, and abstracts are not included. |
| `Review1_Extraction_Form.xlsx` | Full-text decisions and extraction codes for every study, with the coding rules. The column "In extraction sample" marks the random 60. |
| `train_FD001.txt` | NASA C-MAPSS FD001 training data used in the experiment. |
| `run_exp.py` | Runs the experiment: 3 datasets × 2 models × 4 attribution methods × 2 metrics × 3 baselines × 10 seeds. |
| `analyze_exp.py` | Summarises the experiment: changes in the winner, top-1 agreement vs sample size, variance decomposition and bootstrap CIs. |
| `exp_results.json` | Summary numbers reported in the paper. |
| `full_pool_scores.csv` | Mean score of each method for every dataset, model, metric, baseline and seed. |
| `stability.csv`, `min_n.csv`, `variance_decomposition.csv` | Intermediate results behind the figures. |
| `results/` | Raw per-example scores for all 60 runs (one `.npz` file per dataset, model and seed), as used in the paper. |
| `rerun_shap051/` | Summary outputs from an independent rerun on macOS with shap 0.51.0. |

## Reproducing the experiment

Data: Breast Cancer Wisconsin and Digits come with scikit-learn. `train_FD001.txt` is the NASA C-MAPSS FD001 training file (Saxena et al., 2008; U.S. government public data), included here for convenience.

```bash
pip install lightgbm shap lime scikit-learn pandas numpy
python -c "import run_exp; run_exp.run_all()"   # about 1 hour on a laptop; writes results/
python analyze_exp.py                            # writes exp_results.json and the CSV files
```

Versions used for the paper: Python 3.13, scikit-learn 1.9.1, LightGBM 4.7.0, shap 0.54.1, lime 0.2.0.1, numpy 2.5.3 and pandas 3.0.5.

We also reran the experiment independently with shap 0.51.0 (Python 3.11 on macOS). That rerun gave the same conclusions. A few counts based on close calls differed by one setting.

On macOS, LightGBM may need OpenMP. If so, install it with `conda install -c conda-forge lightgbm`.

## Sampling

Random sampling used seed 2026 at both stages. In the first stage, a 15.7% stratified sample of records was drawn from each database. In the second stage, 60 of the 114 records that passed title/abstract screening were drawn for full-text review.

## License

Code: MIT. Data files: CC BY 4.0.
