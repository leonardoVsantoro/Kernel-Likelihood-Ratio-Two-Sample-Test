
# Kernel Likelihood Ratio Two-Sample Test

This repository contains code for **Kernel Likelihood Ratio (KLR) Two-Sample Tests**, a statistical method for comparing two samples to determine whether they come from the same distribution.
The approach leverages **kernel embeddings** of distributions to construct **regularized likelihood ratio statistics**. The method automatically selects:
The implementation adaptively selects the kernel bandwith and regularisation ridge.

> ⚠ **Work in progress:**
> This repository is under active development.

---

## 📁 Repository Structure

```
.
|-- requirements.txt
|-- README.md
|
|-- tests/
|   |-- test.py
|
|-- src/
|   |-- twosample/
|   |-- utils/
|   |-- modules/
|
|-- notebooks/
|   |-- pilot.ipynb
```

---


## 📊 Pilot Notebook

A **pilot notebook** is provided for:

* Introducing the **theory** behind the KLR test
* Illustrating the **code structure**
* Providing **example usage** with visualizations

📄 **Location**: `notebooks/pilot.ipynb`


## 🔹 Tests and Simulations

The `tests/` directory contains unit test for the kernel likelihood ratio two-sample test,
comparing KLR(-0) with classical and state of the art methods, in a dummy Gaussian model with scale shift.

```bash
python tests/test.py
```

This will:
* Generate synthetic data from two distributions
* apply KLR test, compare it to known methods (AggMMD, Spec-Reg-MMD, FR, HT, ...)
* output p-value for each test 

---


## 🔹 Contributors

* **Author:** Leonardo Santoro

---

## 🔹 References

* [Santoro, Waghmare and Panaretos (2025) "From Two Sample Testing to Singular Gaussian Discrimination"](https://arxiv.org/abs/2505.04613)
* [Santoro, Waghmare and Panaretos (2025) "Likelihood Ratio Tests via Kernel Embeddings]

---
