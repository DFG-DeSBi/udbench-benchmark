# UDBench

Code and semi-synthetic benchmark suite for the paper:

> **A Unified Risk View of Uncertainty: Posterior Risk for Disentanglement and Evaluation Beyond Proxies**
> Frieder Wizgall, Georg Tirpitz, Moritz Seiler, Kerstin Ritter, Bálint Mucsányi
> arXiv:2608.05995 — https://arxiv.org/abs/2608.05995

## Overview

Uncertainty quantification methods are usually evaluated through *proxies* — OOD
detection, selective prediction, calibration curves — because the quantity they
claim to estimate is not observable on real data. This repository takes the
other route.

The paper defines uncertainty as **pointwise posterior risk**: the expected loss
of a predictor under the distribution of plausible ground-truth functions given
the data. On semi-synthetic datasets, where the ground-truth function
distribution is a Gaussian process fit to a real UCI/OpenML dataset, that
posterior risk is available in closed form. Oracle epistemic and aleatoric
uncertainty can therefore be computed directly and compared against what a model
reports, with no proxy task in between.

Two findings the benchmark surfaces: accurate prediction does not imply reliable
uncertainty disentanglement, and methods separate meaningfully once measured this
way — but are sensitive to dataset and modeling choices.

## Repository layout

| Path | Contents |
| --- | --- |
| [udbench/datasets/](udbench/datasets/) | `DataSet` — semi-synthetic dataset construction, presets, samplers, kernels |
| [udbench/semi_synth_datasets/preset_outputs/](udbench/semi_synth_datasets/preset_outputs/) | Exported GP preset bundles (`preset.json`), one per dataset |
| [udbench/ground_truth_UD/](udbench/ground_truth_UD/) | Oracle uncertainty: `GroundTruthFnDist`, `GTDisentangle` |
| [udbench/benchmarking/](udbench/benchmarking/) | `UDBench` evaluation harness and metrics |
| [udbench/models/](udbench/models/) | The 14 benchmarked uncertainty-decomposition regressors |
| [udbench/tuning/](udbench/tuning/) | W&B sweep machinery, search spaces, tuning objectives |
| [udbench/experiments/](udbench/experiments/) | Experiment config, dataset registry, runner, CSV I/O |
| [experiments/](experiments/) | Six runnable experiment entry points (see [EXPERIMENTS.txt](experiments/EXPERIMENTS.txt)) |
| [UDBoost/](UDBoost/) | Vendored NGBoost/UDBoost sources used by the tree-based wrappers |

## Installation

Python 3.10+. JAX runs on CPU by default; no GPU extras are required.

```bash
git clone git@github.com:WizgallF/udbench-benchmark.git
cd udbench-benchmark
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` performs an editable install with the `benchmarks` and `test`
extras (`-e .[benchmarks,test]`). The tree-based wrappers use the bundled
`UDBoost/` sources rather than a PyPI wheel.

## Quick start

```python
from udbench.datasets import DataSet
from udbench.benchmarking.Benchmark import UDBench
from udbench.models import get_model_class

# Semi-synthetic dataset: real features, GP ground-truth function, known noise
ds = DataSet.from_preset("abalone", num_observations=750, eval_size=5000, key=42)

model = get_model_class("TabularBNNDeepRegressor")()
out = model.ud_fit_predict(ds.X_obs, ds.y_obs, ds.X_eval)

bench = UDBench(
    dataset=ds,
    y_pred=out["y_pred"],
    pred_total_uncertainty=out["total_uncertainty"],
    pred_epistemic=out["epistemic_uncertainty"],
    pred_aleatoric=out["aleatoric_uncertainty"],
)
results = bench.evaluate_ud()
```

`evaluate_ud` scores predicted total / epistemic / aleatoric uncertainty against
the oracle targets using MSE, Spearman ρ and Pearson r, plus top-{1,5,10,20}%
ranking overlap.

## Datasets

Each preset is a real tabular dataset with a Gaussian process fit to it, turning
it into a semi-synthetic problem with a known ground-truth function
distribution. Presets are discovered from `preset.json` bundles under
[udbench/semi_synth_datasets/preset_outputs/](udbench/semi_synth_datasets/preset_outputs/).

**Development suite** (used for the ablations, exp01–exp05):
`abalone`, `airfoil_self_noise`, `concrete_compressive_strength`,
`cpu_activity`, `qsar_fish_toxicity`, `seoul_bike_sharing_demand`,
`kin8nm:kin8nm_matern32`.

**Validation suite** (main results, exp06):
`health_insurance`, `kings_county_house_prices`, `diamonds`,
`physicochemical_properties_of_protein_tertiary_structure`, `sarcos`,
`space_ga`, `white_wine_quality`.

Canonical sample sizes per dataset live in
[udbench/experiments/datasets_registry.py](udbench/experiments/datasets_registry.py).
Noise is heteroscedastic by default (a rank ramp along the first principal
component); `noise_mode="homoscedastic"` switches it off.

## Models

Registered in [udbench/models/\_\_init\_\_.py](udbench/models/__init__.py) and
constructed via `get_model_class(name)`:

- **Linear** — `BayesianLinearRegressor`
- **Tree boosting** — `NGBoostNIGRegressor`, `NGBoostBaggingRegressor`,
  `CatBoostPosteriorSamplingRegressor`, `CatBoostKGBRegressor`
- **Deep kernel learning** — `DKLRegressor`
- **Bayesian neural networks (JAX/Flax)** — `TabularBNNBaggingRegressor`,
  `TabularBNNDeepRegressor`, `TabularBNNDropoutRegressor`,
  `TabularBNNLaplaceRegressor`, `TabularBNNFSPLaplaceRegressor`,
  `TabularBNNSWAGRegressor`, `TabularBNNEDLRegressor`, `TabularBNNDEUPRegressor`

Every model implements [`BaseUDRegressor`](udbench/BaseUDRegressor.py), whose
`ud_fit_predict` returns `y_pred`, `total_uncertainty`,
`epistemic_uncertainty` and `aleatoric_uncertainty`. Adding a method to the
benchmark means implementing that interface and adding one registry entry.

## Reproducing the experiments

Run from the repository root. Results are written as CSV into each experiment's
`results/` directory. W&B hyperparameter tuning is on by default — pass
`--no-tune` to use default hyperparameters instead.

```bash
# Main results: all models on the 7 validation datasets
python experiments/exp06_validation_benchmark/run.py

# Ablations on the 7 development datasets
python experiments/exp01_bnn_activation/run.py       # relu vs tanh
python experiments/exp02_tuning_objective/run.py     # val_rmse vs val_nll
python experiments/exp03_bnn_optimizer/run.py        # adam / sgd / soap / muon
python experiments/exp04_bnn_width_ablation/run.py   # width 16…512
python experiments/exp05_bnn_depth_ablation/run.py   # depth 1…6
```

Common flags: `--models`, `--datasets`, `--tune-count`, `--no-tune`, `--seed`.
Full option list in [experiments/EXPERIMENTS.txt](experiments/EXPERIMENTS.txt).

## Citation

```bibtex
@misc{wizgall2026unified,
  title         = {A Unified Risk View of Uncertainty: Posterior Risk for
                   Disentanglement and Evaluation Beyond Proxies},
  author        = {Wizgall, Frieder and Tirpitz, Georg and Seiler, Moritz and
                   Ritter, Kerstin and Mucs{\'a}nyi, B{\'a}lint},
  year          = {2026},
  eprint        = {2608.05995},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2608.05995}
}
```

## License

MIT — see [LICENSE](LICENSE).
