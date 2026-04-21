import numpy as np
import networkx as nx
import ot

def solve_Sub_Quad(cost_matrix, iterations=5):
    """
    Approximates EMD using a Primal-Dual loop with explicit data structures.

     Args:
        cost_matrix (np.ndarray): n x n matrix of pairwise distances.
        iterations (int): Number of primal-dual steps (O(1) in paper).
        
    Returns:
        float: Estimated EMD value.
    """
    
    n = cost_matrix.shape[0]
    u = np.zeros(n)
    v = np.zeros(n) 
    matching = {}

    print(f"--- DEBUG START (n={n}) ---")
    
    for step in range(iterations):
        print(f"\n[ITERATION {step + 1}]")
        
        # 1. Calculate Slack and Eligibility
        slack_matrix = cost_matrix - u[:, np.newaxis] - v[np.newaxis, :]
        eligible_indices = np.where(np.isclose(slack_matrix, 0, atol=1e-5))
        
        eligibility_graph = nx.Graph()
        for i, j in zip(eligible_indices[0], eligible_indices[1]):
            eligibility_graph.add_edge(f"r{i}", f"c{j}")

        print(f" -> Eligible Edges: {len(eligible_indices[0])}")

        # 2. Augmenting Paths
        unmatched_rows = [f"r{i}" for i in range(n) if f"r{i}" not in matching]
        matched_cols = set(matching.values())
        
        matches_this_round = 0
        for r_node in unmatched_rows:
            if r_node in matching: continue # Safety check
            try:
                paths = nx.shortest_path(eligibility_graph, source=r_node)
                for target, path in paths.items():
                    if target.startswith("c") and target not in matched_cols:
                        # Update matching 
                        for k in range(0, len(path) - 1, 2):
                            r_key = path[k]
                            c_node = path[k+1]
                            matching[r_key] = c_node
                            matched_cols.add(c_node)
                        
                        matches_this_round += 1
                        break 
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        
        print(f" -> Matches found this round: {matches_this_round}")
        print(f" -> Total matching size: {len(matching)}/{n}")

        # 3. Identify Reachable Nodes for Dual Update
        reachable = set()
        unmatched_now = [f"r{i}" for i in range(n) if f"r{i}" not in matching]
        
        if not unmatched_now:
            print(" -> Perfect matching achieved. Breaking.")
            break

        for r_node in unmatched_now:
            reachable.add(r_node)
            if eligibility_graph.has_node(r_node):
                # Descendants in an undirected graph are just the connected component
                reachable.update(nx.node_connected_component(eligibility_graph, r_node))

        reachable_r = [node for node in reachable if node.startswith("r")]
        reachable_c = [node for node in reachable if node.startswith("c")]
        print(f" -> Reachable Set: {len(reachable_r)} rows, {len(reachable_c)} cols")

        # 4. Calculate Delta
        r_indices = [int(node[1:]) for node in reachable_r]
        # Unreachable columns are those NOT in the reachable set
        c_indices_unreachable = [j for j in range(n) if f"c{j}" not in reachable]
        
        if r_indices and c_indices_unreachable:
            relevant_slacks = slack_matrix[np.ix_(r_indices, c_indices_unreachable)]
            
            if relevant_slacks.size > 0:
                current_delta = np.min(relevant_slacks)
                print(f" -> Delta Calculated: {current_delta:.6f}")
                
                if current_delta <= 1e-9:
                    print(" !!! WARNING: Delta is near zero. Potential infinite loop or precision issue.")

                # 5. Dual Potential Update 
                for i in range(n):
                    if f"r{i}" in reachable:
                        u[i] += current_delta
                    if f"c{i}" in reachable:
                        v[i] -= current_delta
            else:
                print(" -> WARNING: Relevant slacks matrix empty despite candidates.")
        else:
            print(" -> INFO: No dual update possible (No path to unreachable columns).")

    total_cost = sum(cost_matrix[int(r[1:]), int(c[1:])] for r, c in matching.items())
    print(f"\n--- FINAL STATUS ---")
    print(f"Final Matching Size: {len(matching)}/{n}")
    print(f"Total Cost: {total_cost:.4f}")
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
    MAX_VAL = 150
    
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