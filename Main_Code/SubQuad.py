import numpy as np
import networkx as nx
import ot

def solve_Sub_Quad(cost_matrix, iterations=5):
    """
    Approximates EMD using a Primal-Dual loop with explicit data structures.
    Enhanced with verbose logging for debugging.
    """
    n = cost_matrix.shape[0]
    u = np.zeros(n)
    v = np.zeros(n) 
    matching = {}

    print(f"--- Starting Sub-Quad Approximation (n={n}) ---")
    
    for step in range(iterations):
        print(f"\n>>> ITERATION {step + 1} <<<")
        
        # 1. Update Slack and Graph
        slack_matrix = cost_matrix - u[:, np.newaxis] - v[np.newaxis, :]
        eligible_indices = np.where(np.isclose(slack_matrix, 0, atol=1e-8))
        
        num_edges = len(eligible_indices[0])
        print(f"[Step 1] Slack update: {num_edges} edges are now 'eligible' (slack ~ 0)")

        eligibility_graph = nx.DiGraph() # Use Directed!
        eligibility_graph.add_nodes_from([f"r{i}" for i in range(n)] + [f"c{i}" for i in range(n)])
        for i, j in zip(eligible_indices[0], eligible_indices[1]):
            if matching.get(f"r{i}") == f"c{j}":
                # If already matched, we can only go BACKWARDS (C to R)
                eligibility_graph.add_edge(f"c{j}", f"r{i}")
            else:
                # If not matched, we go FORWARDS (R to C)
                eligibility_graph.add_edge(f"r{i}", f"c{j}")

        # 2. Find ALL possible augmentations
        unmatched_rows = [f"r{i}" for i in range(n) if f"r{i}" not in matching]
        # Track who owns which column to make flipping easier
        rev_matching = {v: k for k, v in matching.items()} 
        print(f"[Step 2] Attempting augmentations. Unmatched rows: {len(unmatched_rows)}")
        
        matches_found_this_iter = 0
        for r_start in unmatched_rows:
            if r_start in matching: continue
            try:
                paths = nx.single_source_shortest_path(eligibility_graph, r_start)
                for target, path in paths.items():
                    # Target must be a column AND must be currently UNMATCHED
                    if target.startswith("c") and target not in rev_matching:
                        # AUGMENT: This path flips the matching
                        for k in range(0, len(path)-1, 2):
                            r_node = path[k]
                            c_node = path[k+1]
                            # If this column was matched elsewhere, remove that old match
                            if c_node in rev_matching:
                                old_r = rev_matching[c_node]
                                if old_r in matching: del matching[old_r]
                            
                            matching[r_node] = c_node
                            rev_matching[c_node] = r_node
                            
                        matches_found_this_iter += 1
                        break 
            except nx.NodeNotFound: # Catching the error you just had
                continue
        
        print(f"        Matches added this iteration: {matches_found_this_iter}")
        print(f"        Total current matching size: {len(matching)}/{n}")

        # 3. Reachable set logic
        still_unmatched_rows = [f"r{i}" for i in range(n) if f"r{i}" not in matching]
        if not still_unmatched_rows:
            print("[Step 3] No unmatched rows left. Breaking.")
            break
            
        reachable = set()
        queue = list(still_unmatched_rows)
        reachable.update(queue)
        while queue:
            curr = queue.pop(0)
            for neighbor in eligibility_graph.neighbors(curr):
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)
        
        r_nodes = [node for node in reachable if node.startswith("r")]
        c_nodes = [node for node in reachable if node.startswith("c")]
        print(f"[Step 3] Reachable from unmatched: {len(r_nodes)} rows, {len(c_nodes)} columns")

        # 4. Calculate Delta
        reachable_r = [int(node[1:]) for node in r_nodes]
        unreachable_c = [j for j in range(n) if f"c{j}" not in reachable]
        
        print(f"[Step 4] Checking slacks between {len(reachable_r)} reachable rows and {len(unreachable_c)} unreachable cols")

        if reachable_r and unreachable_c:
            relevant_slacks = slack_matrix[np.ix_(reachable_r, unreachable_c)]
            if relevant_slacks.size > 0:
                current_delta = np.min(relevant_slacks)
                print(f"        Calculated Delta: {current_delta:.6f}")

                # 5. Dual Potential Update 
                for i in range(n):
                    if f"r{i}" in reachable:
                        u[i] += current_delta
                    if f"c{i}" in reachable:
                        v[i] -= current_delta
            else:
                print("        WARNING: Relevant slacks matrix is empty.")
        else:
            print("        INFO: All columns reachable or matching full. No Delta update possible.")

        if len(matching) == n:
            print("!!! Perfect matching achieved early !!!")
            break

    total_cost = sum(cost_matrix[int(r[1:]), int(c[1:])] for r, c in matching.items())
    print(f"\n--- Final Status ---")
    print(f"Matched: {len(matching)}/{n}")
    print(f"Total Cost: {total_cost:.4f}")
    
    if len(matching) < n:
        print(f"DANGER: Algorithm ended with {n - len(matching)} unmatched nodes.")
        
    return total_cost


def solve_pot_emd(cost_matrix):
    """Solves the Exact Earth Mover's Distance using POT (LP solver)."""
    n = cost_matrix.shape[0]
    a, b = np.ones(n) / n, np.ones(n) / n
    emd_cost = ot.emd2(a, b, cost_matrix)
    return emd_cost

if __name__ == "__main__":
    # Parameters
    N = 150  # Size of the matrix
    MAX_VAL = 1530
    
    # Generate random integer cost matrix for easier manual verification
    np.random.seed(34) # For reproducible results
    random_costs = np.random.randint(1, MAX_VAL, size=(N, N)).astype(float)
    
    # Run the solver
    final_emd = solve_Sub_Quad(random_costs, iterations=10)
    actual_emd = solve_pot_emd(random_costs)
    
    print("-" * 30)
    print(f"FINAL ESTIMATED COST: {final_emd/N}")
    print(f"ACTUAL EMD IS: {actual_emd}")
    print("-" * 30)