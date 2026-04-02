import numpy as np
import networkx as nx
import ot

def solve_Sub_Quad(cost_matrix, iterations=5, gamma=0.05, delta = 0.1):
    """
    Approximates EMD using a Primal-Dual loop with explicit data structures.
    
    Args:
        cost_matrix (np.ndarray): n x n matrix of pairwise distances.
        iterations (int): Number of primal-dual steps (O(1) in paper).
        gamma (float): Fraction of allowed outliers (unmatched nodes).
        
    Returns:
        float: Estimated EMD value.
    """
    n = cost_matrix.shape[0]
    u = np.zeros(n)
    v = np.zeros(n) 
    matching = {}

    for _ in range(iterations):
        # Edges (i, j) are eligible if u[i] + v[j] == cost[i, j]
        eligibility_graph = nx.Graph()
        slack_matrix = cost_matrix - u[:, np.newaxis] - v[np.newaxis, :]
        eligible_indices = np.where(np.isclose(slack_matrix, 0, atol=1e-5))
        
        for i, j in zip(eligible_indices[0], eligible_indices[1]):
            eligibility_graph.add_edge(f"r{i}", f"c{j}")

        #Augmenting Paths
        unmatched_rows = [f"r{i}" for i in range(n) if f"r{i}" not in matching]
        matched_cols = set(matching.values())
        
        for r_node in unmatched_rows:
            try:
                path = nx.shortest_path(eligibility_graph, source=r_node)
                target = path[-1]
                if target.startswith("c") and target not in matched_cols:
                    for k in range(0, len(path) - 1, 2):
                        r_idx = int(path[k][1:])
                        c_node = path[k+1]
                        matching[f"r{r_idx}"] = c_node
                        matched_cols.add(c_node)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        #Quasi-Maximal Forest & Update Potentials
        reachable = set()
        for r_node in [f"r{i}" for i in range(n) if f"r{i}" not in matching]:
            if eligibility_graph.has_node(r_node):
                reachable.update(nx.descendants(eligibility_graph, r_node))
                reachable.add(r_node)

        #Calculate Delta
        reachable_r = [int(node[1:]) for node in reachable if node.startswith("r")]
        unreachable_c = [j for j in range(n) if f"c{j}" not in reachable]
        
        if reachable_r and unreachable_c:
            relevant_slacks = slack_matrix[np.ix_(reachable_r, unreachable_c)]
            if relevant_slacks.size > 0:
                delta = np.min(relevant_slacks)

        #Dual Potential Update 
        for i in range(n):
            if f"r{i}" in reachable:
                u[i] += delta  # Increase potential for rows in forest 
            if f"c{i}" in reachable:
                v[i] -= delta  # Decrease potential for columns in forest

        print(f"Edges in Eligibility Graph: {eligibility_graph.number_of_edges()}")
        print(f"Current Matching Size: {len(matching)}")
        

    total_cost = sum(cost_matrix[int(r[1:]), int(c[1:])] for r, c in matching.items())
    return total_cost