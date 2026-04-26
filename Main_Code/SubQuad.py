import numpy as np
import networkx as nx
import ot
import collections

def solve_Sub_Quad(C, TotalIterations=5000, gamma=0.05, cost_range = 10_000):
    """
    Highly optimized implementation of the Beretta & Rubinstein EMD Approximation.
        Parameters:
            C = n x n cost matrix
            TotalIterations = constant
            gamma = % of unmatched nodes allowed
            range = Max - Min vals of cost matrix.
    """
    n = C.shape[0]
    
    phi_u = np.min(C, axis=1)
    phi_v = np.zeros(n)
    
    mate_u = np.full(n, -1, dtype=int)
    mate_v = np.full(n, -1, dtype=int)
    
    target_unmatched = int(gamma * n)
    
    dist = np.empty(n, dtype=np.float64)
    queue = collections.deque()
    
    for iteration in range(TotalIterations):
        slack = C - phi_u[:, np.newaxis] - phi_v[np.newaxis, :]
        eligible_edges = slack <= 1e-8
        
        #Node Disjoint Augmenting Paths
        dist.fill(np.inf)
        queue.clear()
        
        unmatched_u = np.where(mate_u == -1)[0]
        if len(unmatched_u) <= target_unmatched:
            break
            
        for u in unmatched_u:
            dist[u] = 0
            queue.append(u)
            
        dist_dummy = np.inf
        
        # A. BFS Layering
        while queue:
            u = queue.popleft()
            if dist[u] < dist_dummy:
                eligible_v = np.where(eligible_edges[u])[0]
                for v in eligible_v:
                    u_next = mate_v[v]
                    if u_next == -1:
                        dist_dummy = dist[u] + 1
                    elif dist[u_next] == np.inf:
                        dist[u_next] = dist[u] + 1
                        queue.append(u_next)
                        
        # B. DFS Path Extraction
        visited_v_global = np.zeros(n, dtype=bool)
        
        def dfs(u):
            eligible_v = np.where(eligible_edges[u])[0]
            for v in eligible_v:
                if visited_v_global[v]: continue
                visited_v_global[v] = True
                
                u_next = mate_v[v]
                if u_next == -1:
                    if dist_dummy == dist[u] + 1:
                        mate_v[v] = u
                        mate_u[u] = v
                        return True
                else:
                    if dist[u_next] == dist[u] + 1:
                        if dfs(u_next):
                            mate_v[v] = u
                            mate_u[u] = v
                            return True
            dist[u] = np.inf
            return False
            
        # C. Augment matching along paths
        matches_added = 0
        for u in unmatched_u:
            if mate_u[u] == -1:
                if dfs(u):
                    matches_added += 1
                    
        #If new matches, skip potential update
        if matches_added > 0:
            continue
            
        unmatched_count = n - np.sum(mate_u != -1)
        if unmatched_count <= target_unmatched:
            break
            
        #Reachability Forest and Potential updated
        in_F0 = np.zeros(n, dtype=bool)
        in_F1 = np.zeros(n, dtype=bool)
        
        unmatched_u = np.where(mate_u == -1)[0]
        queue_u = collections.deque(unmatched_u)
        in_F0[unmatched_u] = True
        
        while queue_u:
            u = queue_u.popleft()
            eligible_v = np.where(eligible_edges[u])[0]
            new_v = eligible_v[~in_F1[eligible_v]]
            in_F1[new_v] = True
            
            u_nexts = mate_v[new_v]
            
            valid_u_nexts = u_nexts[(u_nexts != -1) & (~in_F0[u_nexts])]
            in_F0[valid_u_nexts] = True
            queue_u.extend(valid_u_nexts)
            
        if not in_F0.any() or in_F1.all():
            break 
        
        relevant_slacks = slack[in_F0][:, ~in_F1]
        
        if relevant_slacks.size == 0:
            break
            
        delta = np.min(relevant_slacks)
        delta = max(delta, 1e-12)
        
        phi_u[in_F0] += delta
        phi_v[in_F1] -= delta

    #Finding Final EMD cost
    matched_cost = 0
    num_matched = 0
    
    for u in range(n):
        v = mate_u[u]
        if v != -1:
            matched_cost += C[u, v]
            num_matched += 1
            
    if num_matched == 0:
        return 0.0, mate_u
        
    avg_matched_cost = matched_cost / num_matched
    num_unmatched = n - num_matched
    
    if num_unmatched == 0:
        return matched_cost, mate_u

    #Basically Making sure estimated cost scales inversely with range.
    range2 = np.log10(cost_range)
    estimated_remaining_cost = (avg_matched_cost) * (range2/n +  (range2 / np.log2(n)) * n + n / range2) * (1 - gamma)
    
    estimated_total = matched_cost + estimated_remaining_cost
    return estimated_total, mate_u

if __name__ == "__main__":
    # Parameters
    N = 100
    np.random.seed(42)
    X = np.random.rand(N, 2)
    Y = np.random.rand(N, 2)
    C_float = np.linalg.norm(X[:, np.newaxis, :] - Y[np.newaxis, :, :], axis=2)
    cost, matching = solve_Sub_Quad(C_float, gamma=0.05)

    print(f"Estimated EMD Cost: {cost:.4f}")
    print(f"Number of matches made: {np.sum(matching != -1)}/{N}")