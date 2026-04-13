# CA World Model

Research codebase for studying whether learned world models can discover effective dynamical variables in cellular automata.

The main question here is not just whether a model can predict the next frame, but whether it can learn a latent state that is:

- compressed,
- approximately closed under time evolution,
- useful for long-horizon rollout,
- and at least partly interpretable through observables or moving structures.

This repository is built around four complementary views of CA dynamics:

- `exact rule rollout`: the ground-truth simulator.
- `pixel predictor`: a direct `x_t -> x_{t+1}` predictor in observation space.
- `dense world model`: an encoder `E`, latent dynamics `F`, and decoder `D`.
- `object world model`: a lightweight slot-style latent model for 1D CA.

## Recent Update

Update on April 13, 2026:

- The dense latent transition now exposes more explicit dynamics controls, including configurable kernel size, residual step size `alpha`, optional local channel normalization, final-layer initialization scaling, and delta clamping.
- The spatial latent dynamics code now supports per-position channel normalization with either local layer normalization or RMS-style normalization for both 1D and 2D models.
- The repo includes a paired baseline-vs-CA-biased comparison runner for summarizing how architectural inductive bias affects one-step accuracy, rollout quality, and closure metrics across multiple 1D rules.
- The README now documents the repo as a full workflow rather than just a loose collection of scripts.

## What Is Implemented

- Fully working 1D elementary CA pipeline with Rules 30, 54, 110, and 184.
- Lightweight 2D Life-like CA support with Conway's Life (`B3/S23`) as the default example.
- Config-driven generation of train, validation, and test trajectories.
- Pixel baseline, dense latent world model, and a modest slot-based 1D object model.
- Training, evaluation, latent visualization, rollout comparison, and probe analysis scripts.
- Tools for studying:
  - long-horizon rollout error,
  - Hamming distance versus horizon,
  - density and domain-wall observable error,
  - latent closure error,
  - linear probes from latent states to observables,
  - event-focused or hand-crafted seeds for structured rules such as 54 and 110.

## Research Framing

The working hypothesis is that some CA admit a more stable or structured latent representation than raw cell space suggests. This repo lets us test questions like:

- Can a learned latent rollout stay accurate longer than a direct pixel-space rollout?
- Is the latent approximately closed, meaning `F^tau(E(x_t))` stays near `E(x_{t+tau})`?
- Do simple probes recover meaningful observables such as density or domain-wall density?
- For structured rules like 54 and 110, does the latent appear to track localized moving patterns rather than only memorizing the next frame?
- How much do architectural biases in the latent dynamics help on hard long-horizon rules?

Rule 184 is usually the easiest place to start, Rule 54 introduces richer moving structures, Rule 110 is the hardest structured rule in this repo, and Rule 30 is useful as a more chaotic stress test.

## Installation

The project targets Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Core dependencies include PyTorch, NumPy, SciPy, pandas, matplotlib, scikit-learn, PyYAML, and Jupyter.

## Repository Layout

```text
CA_world_model/
  configs/
    data/
    experiment/
    model/
    train/
  data/
  notebooks/
  outputs/
  src/
    analysis/
    ca/
    models/
    scripts/
    training/
    utils/
```

## Main Components

### Cellular automata

- `src/ca/elementary_1d.py`: Wolfram elementary CA with periodic boundaries.
- `src/ca/life_like_2d.py`: Life-like 2D CA with `B/S` rule parsing.
- `src/ca/trajectory.py`: rollout generation and initial-condition sampling.
- `src/ca/observables.py`: density, domain walls, and simple event signals.
- `src/ca/visualization.py`: exact and predicted trajectory plots plus 2D GIF helpers.

### Models

- `pixel predictor`: direct convolutional next-step baseline in state space.
- `dense world model`: encoder, latent transition, and decoder with spatial or vector latent options.
- `object world model`: simple slot pooling, slot dynamics, and broadcast decoding for 1D experiments.

### Training and evaluation

- `src/scripts/generate_data.py`: generate train, val, and test splits from a config.
- `src/scripts/train.py`: train a model and emit a comparison image for the first test sample.
- `src/scripts/evaluate.py`: compute one-step and rollout metrics from a checkpoint.
- `src/scripts/compare_to_rule.py`: compare exact CA evolution against a trained model rollout.
- `src/scripts/visualize_latent.py`: inspect learned latent geometry.
- `src/scripts/run_probe_analysis.py`: fit simple probes from latent states to observables.
- `src/scripts/plot_results.py`: plot saved metrics from a run directory.
- `src/scripts/run_ca_biased_comparison.py`: run paired baseline and CA-biased sweeps and aggregate metrics across rules.

## Data and Outputs

The repository intentionally does not track generated datasets, most experiment outputs, or the separate Overleaf/LaTeX workspace.

- `data/*.npz`, `data/*.csv`, `data/*.json`, and `data/*.pt` are ignored.
- `outputs/*` is ignored except for `outputs/.gitkeep`.
- The nested Overleaf repository folder is excluded from Git.

That means a fresh clone gives you the code and configs, but you will need to generate data locally before training.

## End-to-End Workflow

### 1. Generate data

```bash
python -m src.scripts.generate_data --config configs/experiment/rule184_dense.yaml
python -m src.scripts.generate_data --config configs/experiment/rule54_pixel.yaml
```

This creates split files such as:

- `data/rule184_dense_train.npz`
- `data/rule184_dense_val.npz`
- `data/rule184_dense_test.npz`

### 2. Train a model

```bash
python -m src.scripts.train --config configs/experiment/rule184_dense.yaml
python -m src.scripts.train --config configs/experiment/rule54_pixel.yaml
```

By default, training writes a timestamped run directory under `outputs/`, for example:

- `outputs/rule184_dense_20260413_123456/`

Typical artifacts include:

- `best.ckpt`
- `last.ckpt`
- `metrics.csv`
- `compare/compare_sample_0.png`

### 3. Evaluate a checkpoint

```bash
python -m src.scripts.evaluate \
  --checkpoint outputs/<run_dir>/best.ckpt \
  --config configs/experiment/rule184_dense.yaml
```

Useful evaluation flags:

- `--horizon`: rollout horizon for multi-step evaluation.
- `--feedback_mode hard|soft`: choose thresholded or soft feedback during evaluation.
- `--rollout_mode latent|reencode`: roll out directly in latent space or re-encode after each predicted frame.

Expected evaluation outputs include:

- `eval/one_step_metrics.csv`
- `eval/rollout_metrics.csv`
- `eval/hamming_vs_horizon.png`

### 4. Compare against exact rule evolution

```bash
python -m src.scripts.compare_to_rule \
  --checkpoint outputs/<run_dir>/best.ckpt \
  --config configs/experiment/rule184_dense.yaml
```

### 5. Inspect latent structure

```bash
python -m src.scripts.visualize_latent \
  --checkpoint outputs/<run_dir>/best.ckpt \
  --config configs/experiment/rule184_dense.yaml

python -m src.scripts.run_probe_analysis \
  --checkpoint outputs/<run_dir>/best.ckpt \
  --config configs/experiment/rule184_dense.yaml
```

### 6. Plot saved metrics

```bash
python -m src.scripts.plot_results --run_dir outputs/<run_dir>
```

## Recommended Starting Path

For a first pass through the project:

1. Generate data for `rule184_dense`.
2. Train the dense world model on Rule 184.
3. Run evaluation and inspect Hamming distance versus rollout horizon.
4. Use `compare_to_rule` to inspect a concrete held-out trajectory.
5. Run latent visualization and probe analysis.
6. Compare the same rule or a harder rule against the pixel baseline.
7. Move on to `rule54_dense`, then `rule110_best_long`, and then the CA-biased comparison sweep.

## Notable Experiment Families

- `rule184_dense`, `rule54_dense`, `rule110_dense`, `rule30_dense`: core dense latent baselines.
- `rule54_pixel`: direct pixel-space baseline.
- `rule54_object`: lightweight object-centric baseline.
- `*_best_long`: stronger long-horizon baseline configs for 1D rules.
- `*_ca_biased_small`: smaller CA-biased latent dynamics variants for paired comparison.
- `rule110_focus_*`: structured Rule 110 runs that emphasize specific event or defect regimes.
- `rule184_latent_rollout_*` and similar configs: targeted rollout and closure experiments.

## Dense Dynamics Notes

The dense world model is the main research baseline in this repo. It supports:

- spatial or vector latents,
- configurable latent dynamics depth,
- configurable dynamics kernel size,
- residual latent update scale through `dynamics_alpha`,
- optional local channel normalization through `dynamics_norm_type`,
- scaled final transition initialization through `dynamics_init_scale`,
- and optional delta clipping through `dynamics_clamp_delta`.

The newer CA-biased small configs use a shallower, more constrained latent transition with stronger inductive bias and a smaller initialization scale. Those runs are intended for controlled comparisons against the more standard dense baseline.

## Comparison Sweep

To compare baseline and CA-biased models across multiple 1D rules:

```bash
python -m src.scripts.run_ca_biased_comparison \
  --output_root outputs/ca_biased_comparison \
  --device cpu
```

This script trains paired runs, saves per-run evaluation files, and writes:

- `outputs/ca_biased_comparison/summary.csv`
- `outputs/ca_biased_comparison/paired_summary.csv`

The paired summary reports deltas for metrics such as one-step BCE, one-step accuracy, final rollout Hamming error, density error, shift-aligned Hamming error, and closure metrics.

## Caveats and Pitfalls

- Rule 110 is substantially harder than Rule 184 or Rule 54. Short-horizon accuracy can still hide long-horizon structural drift.
- Rule 30 can remain closure-poor even when one-step prediction looks decent. That is often scientifically informative rather than just a modeling failure.
- Pixel rollouts can degrade rapidly because thresholding errors compound over time.
- The object model is a pragmatic baseline, not a full slot-attention research stack.
- The 2D pipeline is lighter-weight than the 1D pipeline and is meant more for exploratory experiments than polished large-scale benchmarks.

## Current Scope

Implemented now:

- strong 1D experimentation pipeline,
- dense latent model as the primary effective-theory baseline,
- pixel baseline and modest object-centric 1D baseline,
- config-driven runs, plots, metrics, checkpoints, and notebooks,
- lightweight 2D Life-like support.

Natural next steps:

- richer OOD suites across size, density, and seed families,
- more explicit defect or collision metrics,
- stronger object-centric training and stability,
- larger 2D experiments,
- and richer interpretability tools for Rule 54 and Rule 110 structures.
