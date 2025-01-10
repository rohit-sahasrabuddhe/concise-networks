import numpy as np
from scipy.spatial.distance import jensenshannon as JSD

class Solution():
    def __init__(self, X, W, G, rank):
        self.rank = rank

        Dw = np.diag(W.sum(axis=0))
        Dw_inv = np.diag(1/ W.sum(axis=0)) 

        self.Xhat_out = X @ W @ Dw_inv 
        self.Xhat_in = G @ Dw
        self.Xhat_in /= self.Xhat_in.sum(axis=1)[:, np.newaxis]

        self.Xhat = self.Xhat_out @ self.Xhat_in.T
        self.loss = np.linalg.norm(X - self.Xhat, ord='fro')