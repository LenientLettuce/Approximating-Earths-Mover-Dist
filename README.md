# Efficient Earth Mover's Distance (EMD)
### From Exact Baselines to Truly Subquadratic Approximation

This repository implements and benchmarks various algorithms for computing the **Earth Mover's Distance (EMD)** between discrete probability distributions. Our primary focus is evaluating a novel **truly subquadratic** algorithm ($O(n^{2-\delta})$) against traditional industry-standard methods.

---

## 🚀 Overview
The Earth Mover's Distance is a critical metric in computer vision, NLP, and generative modeling. While exact solutions often hit a quadratic $O(n^2)$ bottleneck, this project explores the implementation of the template algorithm proposed by **Beretta and Rubinstein (STOC 2024)**, which breaks this barrier by allowing a small fraction of unmatched "outliers".

---

## 🛠 Algorithms Implemented

### 1. Traditional Exact Methods (The Ground Truth)
A)
* **Exact EMD:** The usual approach to calculating EMD.
* **Implementation:** Utilizes the `Python Optimal Transport (POT)` library.

B)
* **Hungarian Method / Min-Cost Flow:** Used to solve for a sub-case of EMD where weights are equal.
* **Implementation:** Leverages `scipy.optimize.linear_sum_assignment` for exact optimal matching.

### 2. Modern Optimized Methods (The Industry Standard)
* **Sinkhorn Algorithm:** An entropic regularization approach that is widely used for fast EMD approximation.
* **Implementation:** Utilizes the `Python Optimal Transport (POT)` library.

### 3. Truly Subquadratic Approximation (The Research Focus)
* **Primal-Dual Template:** Our core implementation of Algorithm 1 from Beretta & Rubinstein.
* **Key Features:**
    * Maintains a partial matching and dual potential function.
    * Alternates between finding augmenting paths and growing a reachability forest.
    * Uses explicit data structures (BFS in Eligibility Graphs) for transparency and query complexity measurement.

---

## 📊 Performance Metrics
We evaluate the performance of these algorithms across several dimensions:
* **Query Complexity:** Measuring the number of distance calculations to verify the $O(n^{2-\delta})$ scaling.
* **Accuracy ($\epsilon$-range):** Confirming that our approximation stays within the theoretically proposed additive error range.
* **Outlier Analysis:** Testing how the "outlier fraction" ($\gamma$) affects the tradeoff between speed and precision.

---


**TO RUN**
python3 -m venv .venv
source .venv/bin/activate
pip install -r reqs.txt

Then run the benchmark.ipynb file.