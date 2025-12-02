import numpy as np
from scipy.stats import entropy
from scipy.spatial import cKDTree

def kl_mvn(m0, S0, m1, S1):
    m0 = np.atleast_1d(m0)
    m1 = np.atleast_1d(m1)
    S0 = np.atleast_2d(S0)
    S1 = np.atleast_2d(S1)
    
    N = m0.shape[0]
    
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0
    
    tr_term = np.trace(iS1 @ S0)
    det_term = np.linalg.slogdet(S1)[1] - np.linalg.slogdet(S0)[1]
    quad_term = diff.T @ iS1 @ diff
    
    return 0.5 * (tr_term + quad_term + det_term - N)

def kl_discrete(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    return entropy(p, q)

def kl_knn(x, y, k=5):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    
    n, d = x.shape
    m = y.shape[0]
    
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)
    
    dd_x, _ = tree_x.query(x, k=k+1, p=2)
    rho = dd_x[:, -1]
    
    dd_y, _ = tree_y.query(x, k=k, p=2)
    nu = dd_y[:, -1]
    
    rho = np.maximum(rho, 1e-15)
    nu = np.maximum(nu, 1e-15)
    
    return (d / n) * np.sum(np.log(nu / rho)) + np.log(m / (n - 1))
