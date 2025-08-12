
# Kernel Likelihood Ratio Two-Sample Test

This repository contains code for **Kernel Likelihood Ratio (KLR) Two-Sample Tests**, a statistical method for comparing two samples to determine whether they come from the same distribution.

* [Santoro, Waghmare and Panaretos (2025) "From Two Sample Testing to Singular Gaussian Discrimination"](https://arxiv.org/abs/2505.04613)
* [Santoro, Waghmare and Panaretos (2025) "Likelihood Ratio Tests via Kernel Embeddings](https://arxiv.org/abs/2508.07982)

The approach leverages **kernel embeddings** of distributions to construct **regularized likelihood ratio statistics**. 
The implementation adaptively selects the kernel bandwith and regularisation ridge.

> ⚠ **Work in progress:**
> This repository is under active development.


## 📑 Table of Contents

<!-- - [Kernel Likelihood Ratio Two-Sample Test](#kernel-likelihood-ratio-two-sample-test) -->
- [🧩 Library Components](#-library-components)
    - [🧪 `KernelTwoSampleTest(name)` — Factory Function](#-kerneltwosampletestname--factory-function)
    - [🧫 Test Class Interface](#-test-class-interface)
- [🔹 Tests and Simulations](#-tests-and-simulations)


---


#### 🧩 Library Components
 
The implementation of the Regularized Kernel Likelihood Ratio Test and related kernel-based two-sample tests is structured into modular components:

##### 🧪 `KernelTwoSampleTest(name)` — Factory Function

Creates a **test class** for one of the following methods:
* `'KLR'`: **Kernel Likelihood Ratio**
* `'KLR-0'`: **Centered KLR**

or one of the following known methods:

* `'MMD'`: Maximum Mean Discrepancy
* `'Agg-MMD'`: Aggregated MMD across bandwidths
* `'SpecReg-MMD'`: Spectral Regularized MMD


### 🧫 Sample code

```python
test = KernelTwoSampleTest('KLR')(X, Y)
result = test(num_permutations=500, level=0.05, NUM_CORES=4)
```

**Attributes**:

* `obs_value`: Observed test statistic
* `p_value`: Corrected (Bonferroni) p-value
* `decision`: Binary decision (1 = reject null, 0 = fail to reject)
* `permuted_stats`: Permutation distribution

**Initialization Parameters**:

* `kernel_name`: Distance metric for kernel matrix (e.g. `'sqeuclidean'`)
* `band_factor_ls`: List of bandwidth multipliers (adaptive kernel width)
* `ridge_ls`: List of ridge parameters (regularization)
* `symmetrise`: Whether to replace testing $P =Q$ with $1/2(P+Q) = Q$ -- improves numerical stability
* `project`: Whether to project $S_Y$ onto eigenspace of $S_X$ -- improves numerical stability




```python
import os
os.chdir(os.path.dirname(os.getcwd()))
from src.twosample.ker import KernelTwoSampleTest # Import the KernelTwoSampleTest class 
import numpy as np

# Instantiate the KLR test class
KLR0 = KernelTwoSampleTest('KLR-0')

# Generate synthetic samples
np.random.seed(0)
sample_size = 100
dimension = 500
X = np.random.multivariate_normal(np.zeros(dimension), np.eye(dimension), sample_size)
Y = np.random.multivariate_normal(np.ones(dimension),  np.eye(dimension), sample_size)

# Initialize test 
test = KLR0(X, Y, 
           band_factor_ls=[0.1, 0.5, 1], 
           ridge_ls=[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1],
           kernel_name='sqeuclidean',
           symmetrise=True, project=True)

# Run the test with permutation-based calibration; 500 random permutations, and 8 cores for parallel processing, significance level of 0.05
result = test(num_permutations=500, level=0.05, NUM_CORES=8) 
print(f"P-Value: {result.p_value:.4f}")
print(f"Reject Null: {'Yes' if result.decision == 1 else 'No'}")
```

## 🔹 Tests and Simulations

The `tests/` directory contains unit test for the kernel likelihood ratio two-sample test,
comparing KLR(-0) with classical and state of the art methods, in a dummy Gaussian model with scale shift.

```bash
python tests/test.py
```
This will:
* Generate synthetic data from two (identical) distributions
* apply KLR test, compare it to known methods (AggMMD, Spec-Reg-MMD, FR, HT, ...)
* output p-value for each test 

The `paper/` directory contains reproducible scripts for generating the figures and results reported in the associated papers. For questions, please open an issue or contact me.