import numpy as np
import networkx as nx
import ot
import collections


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

def approximate_emd_truly_subquad(C, max_iterations=5000, gamma=0.05):
    """
    Practical implementation of the Beretta & Rubinstein EMD Approximation.
    Uses delta-jumps to bypass empty integer steps, and early stopping 
    to achieve the gamma-outlier approximation.
    
    Args:
        C (np.ndarray): n x n cost matrix (floating point or integer).
        gamma (float): Fraction of nodes allowed to remain unmatched (outliers).
                       e.g., 0.05 means it stops when 95% of nodes are matched.
        max_iterations: Safety limit for the primal-dual loop.
    """
    n = C.shape[0]
    
    # 1. Initialize potentials. 
    # Setting phi_u to the minimum of each row ensures we instantly 
    # have at least one eligible (tight) edge per row.
    phi_u = np.min(C, axis=1)
    phi_v = np.zeros(n)
    
    mate_u = np.full(n, -1, dtype=int)
    mate_v = np.full(n, -1, dtype=int)
    
    # The approximation condition: stop when only gamma * n vertices remain
    target_unmatched = int(gamma * n)
    
    for iteration in range(max_iterations):
        # ---------------------------------------------------------
        # STEP 1: Find Node-Disjoint Augmenting Paths 
        # (Hopcroft-Karp BFS+DFS on the Eligibility Graph)
        # ---------------------------------------------------------
        dist = np.full(n, np.inf)
        queue = collections.deque()
        
        for u in range(n):
            if mate_u[u] == -1:
                dist[u] = 0
                queue.append(u)
                
        dist_dummy = np.inf
        
        # A. BFS Layering
        while queue:
            u = queue.popleft()
            if dist[u] < dist_dummy:
                slack = C[u, :] - phi_u[u] - phi_v
                eligible_v = np.where(slack <= 1e-8)[0]  # 1e-8 handles float precision
                
                for v in eligible_v:
                    u_next = mate_v[v]
                    if u_next == -1:
                        dist_dummy = dist[u] + 1
                    elif dist[u_next] == np.inf:
                        dist[u_next] = dist[u] + 1
                        queue.append(u_next)
                        
        # B. DFS Path Extraction
        def dfs(u, visited_v):
            slack = C[u, :] - phi_u[u] - phi_v
            eligible_v = np.where(slack <= 1e-8)[0]
            
            for v in eligible_v:
                if v in visited_v: continue
                u_next = mate_v[v]
                
                if u_next == -1:
                    if dist_dummy == dist[u] + 1:
                        mate_v[v] = u
                        mate_u[u] = v
                        return True
                else:
                    if dist[u_next] == dist[u] + 1:
                        visited_v.add(v)
                        if dfs(u_next, visited_v):
                            mate_v[v] = u
                            mate_u[u] = v
                            return True
                        visited_v.remove(v)
            dist[u] = np.inf
            return False
            
        # C. Augment matching along paths
        for u in range(n):
            if mate_u[u] == -1:
                dfs(u, set())
                    
        # Check approximation condition (Theorem 1 / Lemma 4.4)
        unmatched_count = np.sum(mate_u == -1)
        if unmatched_count <= target_unmatched:
            break
            
        # ---------------------------------------------------------
        # STEP 2: Reachability Forest & Dual Update (QMF)
        # ---------------------------------------------------------
        F0 = set() # Reachable rows
        F1 = set() # Reachable columns
        
        queue_u = collections.deque([u for u in range(n) if mate_u[u] == -1])
        for u in queue_u: F0.add(u)
        
        while queue_u:
            u = queue_u.popleft()
            slack = C[u, :] - phi_u[u] - phi_v
            eligible_v = np.where(slack <= 1e-8)[0]
            
            for v in eligible_v:
                if v not in F1:
                    F1.add(v)
                    u_next = mate_v[v]
                    if u_next != -1 and u_next not in F0:
                        F0.add(u_next)
                        queue_u.append(u_next)
                        
        # D. The Delta-Jump 
        # (Fast-forwards the paper's +1 loops)
        F0_list = list(F0)
        V1_minus_F1 = [v for v in range(n) if v not in F1]
        
        if not F0_list or not V1_minus_F1:
            break  # Graph disconnected / trapped
            
        # Find minimum slack bridging the cut
        slacks = C[np.ix_(F0_list, V1_minus_F1)] - phi_u[F0_list, np.newaxis] - phi_v[np.newaxis, V1_minus_F1]
        delta = np.min(slacks)
        
        # Update dual potentials exactly as prescribed
        for u in F0_list: phi_u[u] += max(delta, 1e-12)
        for v in F1: phi_v[v] -= max(delta, 1e-12)

    # ---------------------------------------------------------
    # Extrapolate Final EMD Cost
    # ---------------------------------------------------------
    matched_cost = 0
    num_matched = 0
    
    for u in range(n):
        v = mate_u[u]
        if v != -1:
            matched_cost += C[u, v]
            num_matched += 1
            
    if num_matched == 0:
        return 0.0, mate_u
        
    # Extrapolate the remaining \gamma outliers to estimate total EMD
    estimated_total = (matched_cost / num_matched) * n
    
    return estimated_total, mate_u

if __name__ == "__main__":
    # Parameters
    N = 100
    np.random.seed(42)
    # Generate 100x100 cost matrix with Euclidean distances (for example)
    X = np.random.rand(N, 2)
    Y = np.random.rand(N, 2)
    C_float = np.linalg.norm(X[:, np.newaxis, :] - Y[np.newaxis, :, :], axis=2)

    # gamma=0.0 means it runs until 100% exact matching. 
    # gamma=0.05 will leave up to 5 elements unmatched and extrapolate.
    cost, matching = approximate_emd_truly_subquad(C_float, gamma=0.05)

    print(f"Estimated EMD Cost: {cost:.4f}")
    print(f"Number of matches made: {np.sum(matching != -1)}/{N}")