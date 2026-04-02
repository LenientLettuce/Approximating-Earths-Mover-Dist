import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from CreateGraphs import generate_bipartite_ot_data,generate_dense_ot_data
import ot
import time
import matplotlib.pyplot as plt

def solve_Hungarian_hungarian(cost_matrix):
    """Solves the assignment problem using Hungarian (Hungarian Algorithm)."""
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()
    return total_cost

def solve_pot_emd(cost_matrix):
    """Solves the Exact Earth Mover's Distance using POT (LP solver)."""
    n = cost_matrix.shape[0]
    a, b = np.ones(n) / n, np.ones(n) / n
    emd_cost = ot.emd2(a, b, cost_matrix)
    return emd_cost

def solve_pot_sinkhorn(cost_matrix, reg=0.1):
    """Solves the Entropic Regularized OT (Sinkhorn) using POT."""
    n = cost_matrix.shape[0]
    a, b = np.ones(n) / n, np.ones(n) / n
    sinkhorn_cost = ot.sinkhorn2(a, b, cost_matrix, reg)
    return sinkhorn_cost

if __name__ == "__main__":
    # Test values for N
    n_values = [100, 300, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000]
    results = {'Hungarian': [], 'POT_EMD': [], 'Sinkhorn': []}

    print("Benchmarking...")
    for n in n_values:
        print(f"Testing N = {n}...")
        C = generate_dense_ot_data(n)
        
        t0 = time.perf_counter()
        solve_Hungarian_hungarian(C)
        results['Hungarian'].append(time.perf_counter() - t0)
        
        t0 = time.perf_counter()
        solve_pot_emd(C)
        results['POT_EMD'].append(time.perf_counter() - t0)
        
        t0 = time.perf_counter()
        solve_pot_sinkhorn(C, reg=0.1)
        results['Sinkhorn'].append(time.perf_counter() - t0)

    plt.figure(figsize=(10, 6))
    colors = {'Hungarian': 'royalblue', 'POT_EMD': 'forestgreen', 'Sinkhorn': 'crimson'}
    
    #Math to ensure that constants are not ruining result.
    for method, times in results.items():
        log_n = np.log(n_values)
        log_t = np.log(times)
        k, log_c = np.polyfit(log_n, log_t, 1)
        plt.scatter(n_values, times, color=colors[method], alpha=0.8)
        fit_line = np.exp(log_c) * (np.array(n_values)**k)
        plt.plot(n_values, fit_line, color=colors[method], 
                 label=f'{method} (Slope/Exponent k={k:.2f})', linewidth=2)

    # Set log scales
    plt.xscale('log')
    plt.yscale('log')
    
    plt.title('Log-Log Complexity Analysis ($O(N^k)$)')
    plt.xlabel('Number of Nodes ($N$) [log scale]')
    plt.ylabel('Execution Time (seconds) [log scale]')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()