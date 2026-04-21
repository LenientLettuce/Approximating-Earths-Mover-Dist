import numpy as np
import networkx as nx
import ot

def solve_Sub_Quad(cost_matrix, iterations=5):
    """
    Approximates EMD using a Primal-Dual loop with explicit data structures.
    """
    n = cost_matrix.shape[0]
    u = np.zeros(n)
    v = np.zeros(n) 
    matching = {}

    #print(f"--- Starting Sub-Quad Approximation (n={n}) ---")
    #print(f"Initial Cost Matrix:\n{cost_matrix}\n")

    for step in range(iterations):
        #print(f"=== Iteration {step + 1} ===")
        
        # 1. Calculate Slack and Eligibility
        slack_matrix = cost_matrix - u[:, np.newaxis] - v[np.newaxis, :]
        eligible_indices = np.where(np.isclose(slack_matrix, 0, atol=1e-5))
        
        eligibility_graph = nx.Graph()
        for i, j in zip(eligible_indices[0], eligible_indices[1]):
            eligibility_graph.add_edge(f"r{i}", f"c{j}")

        #print(f"Eligible Edges: {list(eligibility_graph.edges())}")

        # 2. Augmenting Paths
        unmatched_rows = [f"r{i}" for i in range(n) if f"r{i}" not in matching]
        matched_cols = set(matching.values())
        
        found_augmentation = False
        for r_node in unmatched_rows:
            try:
                # Find path in eligibility graph to an unmatched column
                paths = nx.shortest_path(eligibility_graph, source=r_node)
                #print(f"Avaialble paths are {paths}")
                for target, path in paths.items():
                    # If the destination is a column AND it's not currently matched
                    if target.startswith("c") and target not in matched_cols:
                        #print(f"  [+] Found Augmenting Path: {path}")
                        
                        # Update matching using this path
                        # Standard augmenting path: flip matched/unmatched edges
                        for k in range(0, len(path) - 1, 2):
                            r_idx = int(path[k][1:])
                            c_node = path[k+1]
                            matching[f"r{r_idx}"] = c_node
                            matched_cols.add(c_node)
                        
                        found_augmentation = True
                        break # Move to the next unmatched row
                if target.startswith("c") and target not in matched_cols:
                    # Update matching (Simplified augmentation)
                    for k in range(0, len(path) - 1, 2):
                        r_idx = int(path[k][1:])
                        c_node = path[k+1]
                        matching[f"r{r_idx}"] = c_node
                        matched_cols.add(c_node)
                    found_augmentation = True
                    #print(f"  [+] Augmented matching via path: {path}")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        # 3. Identify Reachable Nodes for Dual Update
        reachable = set()
        unmatched_now = [f"r{i}" for i in range(n) if f"r{i}" not in matching]
        for r_node in unmatched_now:
            if eligibility_graph.has_node(r_node):
                reachable.update(nx.descendants(eligibility_graph, r_node))
                reachable.add(r_node)

        for r_node in unmatched_now:
            reachable.add(r_node)
            if eligibility_graph.has_node(r_node):
                reachable.update(nx.descendants(eligibility_graph, r_node))

        # 4. Calculate Delta (Minimum Slack to bring new edge into eligibility)
        reachable_r = [int(node[1:]) for node in reachable if node.startswith("r")]
        unreachable_c = [j for j in range(n) if f"c{j}" not in reachable]
        
        current_delta = 0
        if reachable_r and unreachable_c:
            relevant_slacks = slack_matrix[np.ix_(reachable_r, unreachable_c)]
            if relevant_slacks.size > 0:
                current_delta = np.min(relevant_slacks)
                #print(f"  [*] Calculated Delta: {current_delta:.4f}")

                # 5. Dual Potential Update 
                for i in range(n):
                    if f"r{i}" in reachable:
                        u[i] += current_delta
                    if f"c{i}" in reachable:
                        v[i] -= current_delta
                #print(f"  [*] Updated Potentials: u={u}, v={v}")
        else:
            print("  [*] No Delta update possible (all reachable or matching full).")

        #print(f"End of Iteration {step+1}: Matching Size = {len(matching)}\n")
        if len(matching) == n:
            #print("Perfect matching found early!")
            break

    total_cost = sum(cost_matrix[int(r[1:]), int(c[1:])] for r, c in matching.items())
    #print(f"Final Matching: {matching}")
    return total_cost

def solve_pot_emd(cost_matrix):
    """Solves the Exact Earth Mover's Distance using POT (LP solver)."""
    n = cost_matrix.shape[0]
    a, b = np.ones(n) / n, np.ones(n) / n
    emd_cost = ot.emd2(a, b, cost_matrix)
    return emd_cost

if __name__ == "__main__":
    # Parameters
    N = 100  # Size of the matrix
    MAX_VAL = 10
    
    # Generate random integer cost matrix for easier manual verification
    np.random.seed(42) # For reproducible results
    random_costs = np.random.randint(1, MAX_VAL, size=(N, N)).astype(float)
    
    # Run the solver
    final_emd = solve_Sub_Quad(random_costs, iterations=10)
    actual_emd = solve_pot_emd(random_costs)
    
    print("-" * 30)
    print(f"FINAL ESTIMATED COST: {final_emd/N}")
    print(f"ACTUAL EMD IS: {actual_emd}")
    print("-" * 30)