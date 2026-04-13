# CA World Model

Research codebase for testing whether learned world models can discover effective dynamical variables in cellular automata.

The central question in this repo is not only whether a model can do next-step prediction, but whether it can learn a latent state that is:

- compressed,
- approximately closed under time evolution,
- useful for long-horizon prediction,
- and at least partly interpretable through observables or moving structures.

This repository is built to compare four viewpoints on CA dynamics:

- `exact rule rollout`: the ground-truth simulator.
- `pixel predictor`: a direct `x_t -> x_{t+1}` predictor in state space.
- `dense world model`: an encoder `E`, latent dynamics `F`, and decoder `D`.
- `object world model`: a simple slot/particle-style latent model for 1D CA.

## Implemented

- Fully working 1D elementary CA pipeline with Rules 184, 54, 110, and 30.
- Lightweight 2D Life-like CA support with Conway's Life (`B3/S23`) by default.
- Dataset generation for train/val/test trajectories with metadata and simple OOD density shifts.
- Pixel baseline, dense latent model, and modest slot-based 1D object model.
- Config-driven training, evaluation, rollout comparison, latent visualization, and probe analysis.
- Analysis tools for:
  - long-horizon rollout error,
  - Hamming distance vs horizon,
  - density and domain-wall observable error,
  - latent closure error,
  - linear probes from latent states to observables,
  - event-focused handcrafted seeds for Rules 54 and 110.

## What "Effective Dynamics" Means Here

The dense and object-centric models try to learn latent variables that behave like effective dynamical variables. In practice, this repo lets you ask:

- Is the latent state smaller or more structured than pixels?
- Can latent rollouts stay predictive for longer than direct pixel rollouts?
- Is the latent approximately closed, meaning `F^tau(E(x_t))` matches `E(x_{t+tau})`?
- Do simple probes recover observables like density or domain-wall density from the latent?
- For structured rules like 54 or 110, does the model appear to track moving localized patterns?

## Quickstart

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate data:

```bash
python -m src.scripts.generate_data --config configs/experiment/rule184_dense.yaml
python -m src.scripts.generate_data --config configs/experiment/rule54_pixel.yaml
```

Visualize exact CA:

```bash
python -m src.scripts.visualize_ca --config configs/experiment/rule54_dense.yaml --output_dir outputs/rule54_exact
```

Train:

```bash
python -m src.scripts.train --config configs/experiment/rule184_dense.yaml
python -m src.scripts.train --config configs/experiment/rule54_pixel.yaml
```

Evaluate:

```bash
python -m src.scripts.evaluate --checkpoint outputs/<run_dir>/best.ckpt --config configs/experiment/rule184_dense.yaml
python -m src.scripts.evaluate --checkpoint outputs/<run_dir>/best.ckpt --config configs/experiment/rule54_pixel.yaml
```

Compare rollout to exact rule:

```bash
python -m src.scripts.compare_to_rule --checkpoint outputs/<run_dir>/best.ckpt --config configs/experiment/rule184_dense.yaml
```

Latent visualization and probes:

```bash
python -m src.scripts.visualize_latent --checkpoint outputs/<run_dir>/best.ckpt --config configs/experiment/rule184_dense.yaml
python -m src.scripts.run_probe_analysis --checkpoint outputs/<run_dir>/best.ckpt --config configs/experiment/rule184_dense.yaml
```

Plot saved evaluation metrics:

```bash
python -m src.scripts.plot_results --run_dir outputs/<run_dir>
```

## Recommended First Experiment Sequence

1. Generate `rule184_dense` data.
2. Train the dense latent model on Rule 184.
3. Evaluate and inspect rollout error vs horizon.
4. Run `compare_to_rule` on a held-out sample.
5. Run latent visualization and probe analysis.
6. Repeat with `rule54_pixel` and compare pixel-space vs latent-space long-horizon behavior.
7. Move to `rule54_dense`, then `rule110_dense`, then optionally `rule54_object`.

## Example Output Paths

After a typical training run in `outputs/<run_dir>/`, you can expect files such as:

- `outputs/<run_dir>/best.ckpt`
- `outputs/<run_dir>/eval/rollout_metrics.csv`
- `outputs/<run_dir>/eval/hamming_vs_horizon.png`
- `outputs/<run_dir>/compare/compare_sample_0.png`
- `outputs/<run_dir>/latent_viz/latent_pca.png`
- `outputs/<run_dir>/probe/probe_metrics.csv`

## Repo Structure

```text
ca_world_model/
  configs/
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
- `src/ca/trajectory.py`: rollout generation and initial condition samplers.
- `src/ca/observables.py`: density, domain walls, and simple event signals.
- `src/ca/visualization.py`: exact/predicted trajectory plots and 2D GIFs.

### Models

- `pixel predictor`: direct convolutional next-step baseline.
- `dense world model`: `E`, `F`, `D` with spatial or vector latent options.
- `object world model`: simple slot pooling, slot dynamics, and broadcast decode for 1D.

### Analysis

- rollout error curves,
- exact vs predicted trajectory plots,
- latent PCA and t-SNE,
- closure analysis,
- linear probes to observables,
- event-focused seeds for structured 1D rules.

## Implemented Now vs Future Work

Implemented now:

- Strong 1D research pipeline.
- Dense latent model as the main effective-theory baseline.
- Pixel baseline and basic object-centric 1D baseline.
- Config-driven scripts, plots, metrics, checkpoints, and notebooks.
- Lightweight 2D Life support.

Future work:

- richer OOD suites across system size and seed families,
- more stable slot attention and better object discovery,
- explicit collision/event metrics,
- UMAP and richer clustering tools,
- larger-scale 2D experiments,
- more interpretable defect tracking for Rules 54 and 110.

## Likely Pitfalls

- Rule 110 is harder than Rule 184 or Rule 54: local prediction can look good while long-horizon structure drifts quickly.
- Object-centric training is intentionally modest here; slot collapse or diffuse slot usage can happen without careful tuning.
- Pixel rollouts often accumulate thresholding errors rapidly, so short-horizon metrics can be misleading.
- Closure metrics for chaotic rules like Rule 30 may stay poor even when one-step prediction is decent, which is scientifically informative rather than just a modeling failure.

## Notes on the Models

- `pixel predictor`: directly predicts cells in pixel/state space.
- `dense world model`: encodes to latent state, evolves latent state, decodes back to pixels.
- `object world model`: approximates a structured latent with a small set of slots/particles.
- `exact rule rollout`: the simulator baseline and ground truth.

## Caveats

- The 1D pipeline is the most mature part of the repo.
- The object model is a pragmatic baseline, not a full slot-attention research implementation.
- The 2D path is intentionally lighter so laptop experiments remain feasible.
