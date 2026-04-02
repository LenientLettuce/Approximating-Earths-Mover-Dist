# Approximating-Earths-Mover-Dist
### From Exact Baselines to Truly Subquadratic Approximation

[cite_start]This repository implements and benchmarks various algorithms for computing the **Earth Mover's Distance (EMD)** between discrete probability distributions[cite: 9, 31]. [cite_start]Our primary focus is evaluating a novel **truly subquadratic** algorithm ($O(n^{2-\delta})$) against traditional industry-standard methods[cite: 13, 31].

---

## 🚀 Overview
[cite_start]The Earth Mover's Distance is a critical metric in computer vision, NLP, and generative modeling[cite: 11, 24]. [cite_start]While exact solutions often hit a quadratic $O(n^2)$ bottleneck, this project explores the implementation of the template algorithm proposed by **Beretta and Rubinstein (STOC 2024)**, which breaks this barrier by allowing a small fraction of unmatched "outliers"[cite: 11, 13, 15].

---

## 🛠 Algorithms Implemented

### 1. Traditional Exact Methods (The Ground Truth)
* [cite_start]**Hungarian Method / Min-Cost Flow:** Used as our exact baseline to verify precision[cite: 45].
* [cite_start]**Implementation:** Leverages `scipy.optimize.linear_sum_assignment` for $O(n^2)$ performance[cite: 28, 45].

### 2. Modern Optimized Methods (The Industry Standard)
* [cite_start]**Sinkhorn Algorithm:** An entropic regularization approach that is widely used for fast EMD approximation[cite: 19, 48].
* [cite_start]**Implementation:** Utilizes the `Python Optimal Transport (POT)` library[cite: 72].

### 3. Truly Subquadratic Approximation (The Research Focus)
* [cite_start]**Primal-Dual Template:** Our core implementation of Algorithm 1 from Beretta & Rubinstein[cite: 16].
* **Key Features:**
    * [cite_start]Maintains a partial matching and dual potential function[cite: 17].
    * [cite_start]Alternates between finding augmenting paths and growing a reachability forest[cite: 17].
    * [cite_start]Uses explicit data structures (BFS in Eligibility Graphs) for transparency and query complexity measurement[cite: 27, 38].

---

## 📊 Performance Metrics
We evaluate the performance of these algorithms across several dimensions:
* [cite_start]**Query Complexity:** Measuring the number of distance calculations to verify the $O(n^{2-\delta})$ scaling[cite: 51].
* [cite_start]**Accuracy ($\epsilon$-range):** Confirming that our approximation stays within the theoretically proposed additive error range[cite: 53, 54].
* [cite_start]**Outlier Analysis:** Testing how the "outlier fraction" ($\gamma$) affects the tradeoff between speed and precision[cite: 32].

---
