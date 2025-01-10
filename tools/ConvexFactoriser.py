import numpy as np
from sklearn.cluster import KMeans
from .Solution import Solution

def cost_function(X, W, G):
    reconstruction = X @ W @ G.T 
    return np.linalg.norm(x = X - reconstruction, ord='fro')


def initialise_factors(rank, kind='kmeans', **kwargs):
    if kind == 'random':
        n_preds = kwargs['n_preds']
        W = np.random.rand(n_preds, rank)
        G = np.random.rand(n_preds, rank)

    elif kind == 'kmeans':
        off_cluster = kwargs['off_cluster'] if 'off_cluster' in kwargs else 0.2
        X = kwargs['X']
        km = KMeans(n_clusters=rank, n_init=1, tol=1e-2).fit(X.T)
        labels = km.labels_
        cols = []
        for r in range(rank):
            cluster = (labels == r)
            column = cluster + (1 - cluster)*off_cluster
            cols.append(list(column))
        G = np.array(cols).T
        W = G / G.sum(axis=1)[:, np.newaxis]

    return W, G



def iterate(XTX, W, G):
    G = G * np.sqrt((XTX @ W) / np.maximum(G @ W.T @ XTX @ W , 1e-16))
    W = W * np.sqrt((XTX @ G) / np.maximum(XTX @ W @ G.T @ G, 1e-16))
    return W, G



def run(X, rank, initialise, max_iter=1000, check_every=10, rel_tol=0.0001):
    n_succs, n_preds = X.shape
    
    if initialise == 'random':
        W, G = initialise_factors(rank, kind=initialise, n_preds=n_preds)
    elif initialise == 'kmeans':
        W, G = initialise_factors(rank, kind=initialise, X=X, off_cluster=0.2)

    XTX = X.T @ X

    old_cost = cost_function(X, W, G)

    for iter_num in range(1, max_iter+1):
        if iter_num % check_every == 0:
            new_cost = cost_function(X, W, G)
            if abs(1 - new_cost / old_cost) < rel_tol:
                return Solution(X=X, W=W, G=G, rank=rank)
            
            old_cost = new_cost

        W, G = iterate(XTX, W, G)    

    return Solution(X=X, W=W, G=G, rank=rank)