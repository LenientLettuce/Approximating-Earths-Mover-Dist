import networkx as nx
import numpy as np

def generate_bipartite_ot_data(n, edge_prob=0.7):
    """
    Generates a random bipartite graph and a corresponding cost matrix.
    
    Args:
        n (int): Number of nodes in each partition.
        edge_prob (float): Probability of an edge existing between partitions.
        
    Returns:
        G (nx.Graph): The NetworkX bipartite graph object.
        C (np.ndarray): The n x n cost matrix.
    """
    #Random Graph with n nodes on each side
    G = nx.bipartite.random_graph(n, n, edge_prob)
    
    #Assign random distances
    pos = {i: np.random.rand(2) for i in range(2 * n)}
    
    # Initialize Cost Matrix (Rows: Partition U, Cols: Partition V)
    C = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            u = i
            v = j + n
            # Euclidean distance
            dist = np.linalg.norm(pos[u] - pos[v])
            
            #penalty cost
            C[i, j] = dist if G.has_edge(u, v) else 1e3
                
    return G, C

def generate_dense_ot_data(n):
    """Generates a fully connected cost matrix."""
    # Create two sets of points in 2D space
    random_costs = np.random.randint(1, 10000, size=(n, n)).astype(float)
    return random_costs