import numpy as np
from scipy.optimize import linear_sum_assignment
import ot

def solve_Hungarian(cost_matrix):
    """Solves the assignment problem using Hungarian."""
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
    M = cost_matrix / cost_matrix.max()
    sinkhorn_cost = ot.sinkhorn2(a, b, M, reg, 
                                 numItermax=10000, 
                                 stopThr=1e-9, 
                                 method='sinkhorn')
    
    return sinkhorn_cost * cost_matrix.max()