Of course! Here's a polished `README.md` with a clear note that the project is **work in progress**.

---

# Kernel Likelihood Ratio Two-Sample Test

This repository contains code for **Kernel Likelihood Ratio Two-Sample Tests**, a statistical method for comparing two samples to determine whether they come from the same distribution.
The approach utilizes (Gaussian) kernel embeddings to construct likelihood ratio statistics for two-sample testing.
The implementation adaptively selects the kernel bandwith and regularisation ridge.

> ⚠ **Work in progress:**
> This repository is under active development.
> The code may be incomplete, and the API may change frequently.
> Please check back for future updates.

---

## 📁 Repository Structure

```
.
|-- requirements.txt
|-- out/
|   |-- images/
|   |-- data/
|
|-- tests/
|   |-- sims.py
|   |-- plot/
|   |-- models/
|   |-- run/
|
|-- src/
|   |-- TwoSampleTests/
|   |-- utils/
|   |-- modules/
```

---

## 🔹 Tests and Simulations

The `tests/` directory contains simulation code and unit tests for different components of the kernel likelihood ratio two-sample test.

* **sims.py:** simulation routines
* **run/** directory contains `run.py` for executing test scenarios with fast parallelisation
* **models/** directory contains kernel models for likelihood ratio testing

---

## 🔹 Main Components

| ------------------------------------ | ------------------------------------------------------ |
| `src/TwoSampleTests/kernel_tests.py` | main kernel likelihood ratio two-sample test           |
| ------------------------------------ | ------------------------------------------------------ |
---

## 🔹 How to Run

To execute a two-sample test:

```bash
python tests/run/run.py
```

This will:
* generate samples from two distributions
* apply kernel likelihood ratio two-sample test, compare it to known methods (AggMMD, Spec-Reg-MMD, FR, HT, Energy, ...)
* output average rejection probability

---


## 🔹 Contributors

* **Author:** Leonardo Santoro
* **Contributors:** Charles Dufur

---

## 🔹 References

* [Santoro, Waghmare and Panaretos (2025) "From Two Sample Testing to Singular Gaussian Discrimination"](https://arxiv.org/abs/2505.04613)
* [Santoro, Waghmare and Panaretos (2025) "Likelihood Ratio Tests via Kernel Embeddings]

---
