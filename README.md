
# Kernel Likelihood Ratio Two-Sample Test

This repository contains code to reproduce the experiments in the following papers:

* [Santoro, Waghmare and Panaretos (2025) "From Two Sample Testing to Singular Gaussian Discrimination"](https://arxiv.org/abs/2505.04613)
* [Santoro, Waghmare and Panaretos (2025) "Likelihood Ratio Tests via Kernel Embeddings"](https://arxiv.org/abs/2508.07982)

The method constructs **regularized likelihood ratio statistics** via **kernel embeddings** of distributions, with adaptive selection of kernel bandwidth and regularization ridge.

---

## Repository Structure

```
src/
  sims/       # Simulation experiments (one script per setting)
  real/       # Real-data experiments (MNIST, CIFAR-10)
  utils/      # Shared utilities, test implementations, model classes
out/
  sims/       # Output CSVs for simulation results
  real/       # Output CSVs for real-data results
  plots/      # Figures
  tables/     # Tables
cluster/      # SLURM job scripts for running experiments on a cluster
```

---

## Reproducing the Experiments

### Dependencies

Install requirements with:

```bash
pip install -r requirements.txt
```

### Simulations

Each script in `src/sims/` corresponds to one simulation setting from the paper:

| Script | Setting |
|--------|---------|
| `GaussianSparseMeanShift.py` | Gaussian sparse mean shift |
| `LaplaceSparseMeanShift.py` | Laplace sparse mean shift |
| `GaussianSpikedCovariance.py` | Gaussian spiked covariance |
| `EquiCorrelationGaussian.py` | Equi-correlation Gaussian |
| `DecreasingCorrelationGaussian.py` | Decreasing correlation Gaussian |
| `GaussianMixture.py` | Gaussian mixture |
| `ConcentricSpheres.py` | Concentric spheres |
| `UniformThinHypercube.py` | Uniform thin hypercube |

Run a single simulation locally:

```bash
python src/sims/GaussianSparseMeanShift.py
```

To run all simulations on a SLURM cluster:

```bash
bash cluster/submit_all.sh
```

Results are saved as CSV files in `out/sims/`.

### Real-Data Experiments

```bash
python src/real/mnist.py
python src/real/cifar.py
```

Results are saved in `out/real/`.

### Generating Plots and Tables

```bash
python src/utils/plot.py
```

---

For questions, please open an issue or contact the authors.
