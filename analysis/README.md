# Analysis Suite

This folder adds a clean, reusable evaluation layer on top of the existing `src/` training and loading code. It does not replace the training framework; it reuses the saved configs, datasets, checkpoints, rollout helpers, and model builders already in the repo.

## Files

- `run_all.py`: CLI entrypoint that discovers runs or analyzes explicit config/checkpoint pairs.
- `sanity_checks.py`: rollout sanity checks, baseline comparison, OOD density sweeps, and lattice-size generalization.
- `latent_viz.py`: representative latent-channel plots plus PCA/optional UMAP embeddings.
- `latent_probes.py`: linear probes, latent locality probe, and the Rule 184 conservation analysis.
- `latent_discovery.py`: unsupervised symbolic, conserved-quantity, slow-variable, and continuity-equation discovery on latent states, plus post-hoc validation against known CA labels.
- `baselines.py`: copy-state, majority-local, fitted rule-table, and cached learned CNN baselines.
- `plotting.py`: small plotting helpers with a consistent figure style.
- `utils.py`: run discovery, model/data loading, latent extraction, and trajectory generation helpers.
- `outputs/`: standardized analysis outputs.

## Run It

Analyze one known run:

```bash
python3 analysis/run_all.py \
  --config outputs/rule184_dense_retrain_fast/config.json \
  --checkpoint outputs/rule184_dense_retrain_fast/best.ckpt \
  --rules 184
```

Analyze a config/checkpoint pair from the experiment configs:

```bash
python3 analysis/run_all.py \
  --config configs/experiment/rule184_dense_long.yaml \
  --checkpoint outputs/rule184_dense_long_20260410_114543/best.ckpt \
  --rules 184
```

Discover all `best.ckpt` files under `outputs/` and filter to selected rules:

```bash
python3 analysis/run_all.py --rules 30 54 110 184
```

Useful flags:

- `--search_root`: where discovery mode looks for saved runs.
- `--data_dir`: dataset root, default `data`.
- `--output_root`: analysis output root, default `analysis/outputs`.
- `--max_samples`: cap the number of trajectories used per run for faster iteration.
- `--probe_points`: cap the number of latent position samples used for embedding/probe jobs.
- `--device`: `cpu`, `cuda`, etc.

Run the unsupervised latent discovery module across the default rules and all discovered checkpoints:

```bash
python3 analysis/latent_discovery.py --rules 30 54 110 184
```

Or target a single checkpoint:

```bash
python3 analysis/latent_discovery.py \
  --checkpoint outputs/smoke_rule184_dense_small/best.ckpt \
  --max_samples 8 \
  --cluster_K 2 4 \
  --radii 0 1 \
  --fit_linear
```

## Output Layout

Results are written to:

```text
analysis/outputs/{rule}/{model_name}/{run_name}/
```

Typical subfolders:

- `sanity_checks/`
  - `rollout_sanity_summary.csv`
  - `accuracy_vs_step.png`
  - `hamming_vs_step.png`
- `baseline_comparison/`
  - `baseline_comparison_summary.csv`
  - baseline-vs-model rollout plots
- `ood_density/`
  - per-density rollout summaries and plots
  - for Rule 184, particle-count and density-over-time plots
- `lattice_generalization/`
  - `L=128 -> L=256` rollout results when the model supports variable length
- `latent_viz/`
  - representative spacetime + latent-channel figure
- `latent_embeddings/`
  - PCA plots and optional UMAP plots
- `linear_probes/`
  - `linear_probe_results.csv`
- `locality_probe/`
  - `locality_probe_results.csv`
  - radius-vs-error plot
- `rule184_conservation/`
  - particle-count rollout plots
  - latent conserved-quantity fit summary

There is also a top-level manifest at `analysis/outputs/analysis_manifest.csv`.

The latent-discovery runner writes to:

```text
analysis/outputs/{rule}/{model_name}/latent_discovery/{run_name}/
```

with subfolders for:

- `symbolic/`
- `conserved_quantities/`
- `slow_variables/`
- `continuity_equation/`
- `posthoc_validation/`

## What The Main Outputs Mean

- `accuracy_vs_step.png`: mean cellwise rollout accuracy at horizons `[1, 2, 4, 8, 16, 30, 60]`.
- `hamming_vs_step.png`: mean rollout Hamming error at the same horizons.
- `baseline_comparison_summary.csv`: same rollout metrics for the analyzed model and the four requested baselines.
- `ood_density_summary.csv`: rollout metrics after changing the Bernoulli initial density away from the training setup.
- `latent_channels.png`: one representative trajectory showing exact spacetime, predicted spacetime, error map, and a few high-variance latent channels.
- `embedding_points.csv`: low-dimensional latent coordinates plus labels such as center cell, neighborhood class, and next cell.
- `linear_probe_results.csv`: simple probe scores for what the latent linearly reveals.
- `locality_probe_results.csv`: how well `z_{t+1,i}` can be predicted from a radius-`r` latent neighborhood.
- `rule184_conservation_summary.csv`: how well the model and a linear latent quantity preserve particle number in Rule 184.

## Notes And Graceful Skips

- Latent plots and probes currently target 1D dense spatial latents. Non-spatial or slot-based models are skipped with a short reason file instead of failing.
- Lattice-size generalization only runs for models that can naturally handle arbitrary 1D lengths and only when the training size is `128`.
- UMAP plots are only produced if `umap-learn` is installed.
- The learned CNN baseline is trained once per dataset prefix and cached under the corresponding analysis output tree.
