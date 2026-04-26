import networkx as nx
import numpy as np
from Main_Code import CreateGraphs, BaseLine, SubQuad
import time
import matplotlib.pyplot as plt

def main():
    n_values = [100, 300, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 7500, 10_000, 15_000]
    #n_values = [100, 300, 500, 1000, 1500, 2000, 2500]
    # We track both runtime and the computed cost to evaluate accuracy
    results = {'Hungarian': [], 'POT_EMD': [], 'Sinkhorn': [], 'SubQuad': []}
    accuracy = {'Sinkhorn': [], 'SubQuad': []}

    print("Benchmarking Runtime and Accuracy...")
    for n in n_values:
        print(f"Testing N = {n}...")
        C = CreateGraphs.generate_dense_ot_data(n)
        
        t0 = time.perf_counter()
        true_cost = BaseLine.solve_pot_emd(C) 
        results['POT_EMD'].append(time.perf_counter() - t0)
        
        t0 = time.perf_counter()
        BaseLine.solve_Hungarian(C)
        results['Hungarian'].append(time.perf_counter() - t0)
        
        t0 = time.perf_counter()
        sink_cost = BaseLine.solve_pot_sinkhorn(C, reg=0.0003)
        results['Sinkhorn'].append(time.perf_counter() - t0)
        # Relative Error: |approx - true| / true
        accuracy['Sinkhorn'].append(abs((sink_cost) - true_cost) / true_cost)

        t0 = time.perf_counter()
        sub_quad_cost = SubQuad.solve_Sub_Quad(C, 30, 0.05)
        print(sub_quad_cost[0]/n)
        print(true_cost)
        results['SubQuad'].append(time.perf_counter() - t0)
        accuracy['SubQuad'].append(abs((sub_quad_cost[0]/n) - true_cost) / true_cost)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    colors = {'Hungarian': 'royalblue', 'POT_EMD': 'forestgreen', 'Sinkhorn': 'crimson', 'SubQuad': 'orange'}

    for method, times in results.items():
        log_n = np.log(n_values)
        log_t = np.log(times)
        k, log_c = np.polyfit(log_n, log_t, 1)
        plt.scatter(n_values, times, color=colors[method], alpha=0.8)
        fit_line = np.exp(log_c) * (np.array(n_values)**k)
        plt.plot(n_values, fit_line, color=colors[method], 
                 label=f'{method} (k={k:.2f})')

    plt.xscale('log')
    plt.yscale('log')
    plt.title('Runtime Complexity $O(N^k)$')
    plt.xlabel('N (log)')
    plt.ylabel('Time (s) (log)')
    plt.legend()
    plt.grid(True, which="both", alpha=0.2)

    plt.subplot(1, 2, 2)
    plt.plot(n_values, accuracy['Sinkhorn'], marker='o', color='crimson', label='Sinkhorn Error')
    plt.plot(n_values, accuracy['SubQuad'], marker='s', color='orange', label='SubQuad Error')
    
    plt.title('Relative Error Analysis')
    plt.xlabel('Number of Nodes (N)')
    plt.ylabel('Relative Error (|Approx - True| / True)')
    plt.axhline(y=0.05, color='gray', linestyle='--', label='Target Approx (5%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\nFinal Results for N={n_values[-1]}:")
    print(f"SubQuad Error: {accuracy['SubQuad'][-1]:.4f}")
    print(f"Sinkhorn Error: {accuracy['Sinkhorn'][-1]:.4f}")

main()