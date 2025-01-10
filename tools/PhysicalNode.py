import numpy as np, pandas as pd
from . import ConvexFactoriser
from .Solution import Solution
import scipy.optimize
import copy

class PhysicalNode():
    """ A physical node"""

    def __init__(self):
        """
        Initialise the physical node.

        Parameters:
        id: the ID of the node
        has_trigrams (boolean, default=True): indicate whether the node has any trigrams passing through

        Returns:
        None  
        """
        pass

    @classmethod
    def from_adjacency_matrix(cls, A, id=None, arr_preds=None, arr_succs=None):
        """
        Constructor. Create a PhysicalNode from an adjacency matrix. Sets the following variables:
        id: node id
        A: np.array; adjacency matrix
        T: np.array; MLE for the second order Markov model
        n_preds: int; number of predecessors
        n_succs: int; number of successors
        arr_preds: np.array; array of predecessors
        arr_succs: np.array; array of successors
        has_trigrams: True

        Parameters:
        A: np.array; adjacency matrix
        id: phyiscal node id. The user is responsible for verifying that ids are not repeated in a network.
        arr_preds: np.array; array of predecessors (default None, set to np.arange(n_preds))
        arr_succs: np.array; array of successors (default None, set to np.arange(n_succs))
        """
        pn = cls()
        pn.has_trigrams = True
        pn.id = id
        pn.A = A
        pn.T = A / A.sum(axis=0)
        pn.n_succs, pn.n_preds = A.shape

        if arr_succs is None:
            arr_succs = np.arange(pn.n_succs)
        if arr_preds is None:
            arr_preds = np.arange(pn.n_preds)        
        pn.arr_preds = arr_preds
        pn.arr_succs = arr_succs
        return pn
        
    @classmethod
    def from_M1(cls, id=None, arr_preds=None, arr_succs=None, M1=None):
        pn = cls()
        pn.has_trigrams = False
        pn.id = id
        pn.arr_preds = arr_preds
        pn.arr_succs = arr_succs
        pn.M1 = M1
        return pn


    def create_adjusted_transition_matrix(self, mu=None):
        """
        Regularise T to create X.

        Parameters:
        mu (numeric, default None): The strength of the prior (pseudocount). If None, use leave-one-out-crossvalidation.

        Returns:
        None
        """
        def loo_likelihood(m):
            m = m[0]
            numerator = (self.A - 1 + m*T_M1)         
            return (self.A * np.log( np.maximum( numerator / (pred_matrix + m), 1e-16 ) )).sum() * -1
           

        M1 = self.A.sum(axis=1)
        M1 /= M1.sum()
        T_M1 = np.vstack([M1]*self.n_preds).T
        pred_counts = self.A.sum(axis=0)

        self.mu = mu
        if mu is None:
            pred_matrix = np.vstack([pred_counts-1]*self.n_succs)            
            res = scipy.optimize.minimize(fun=loo_likelihood, x0=[1.], tol=1e-8, bounds=[(1e-16, np.inf)])
            self.mu = res.x[0]

        lam = pred_counts / (pred_counts + self.mu)    
        self.X = lam*self.T + (1.0 - lam)*T_M1


           
    def process(self, max_rank, n_candidates, overlap_threshold=None, **kwargs):
        """
        Factorise the node.

        Parameters:
        max_rank (int)
        n_candidates (int)
        overlap_threshold (float, default None): threshold of flow overlap to stop factorising. If None, do all ranks until max_rank.
        **kwargs: passed onto ConvexFactoriser.run
            initialise (str, default 'kmeans'): the method to initialse the ConvexFactoriser. Options are 'kmeans' and 'random'.
            max_iter (int, default 1000): the maximum number of iterations to look for convergence
            check_every (int, default 10): number of iterations between successive checks for convergence
            rel_tol (float, default 1e-4): relative tolerance of loss in check_every iterations to declare convergence       

        Returns:
        None
        """

        if 'initialise' not in kwargs:
            kwargs['initialise'] = 'kmeans'
        if 'max_iter' not in kwargs:
            kwargs['max_iter'] = 1000
        if 'check_every' not in kwargs:
            kwargs['check_every'] = 10
        if 'rel_tol' not in kwargs:
            kwargs['rel_tol'] = 1e-4

        self.dict_solutions = dict()
        self.dict_flow_overlap = dict()

        # rank=1 is deterministic
        rank1_sol = Solution(
            X = self.X,
            W = np.ones((self.n_preds, 1)), 
            G = np.ones((self.n_preds, 1)), 
            rank = 1
        )
        self.dict_solutions[1] = rank1_sol
        self.dict_flow_overlap[1] = ( (self.A.sum(axis=0) / self.A.sum()) * np.minimum(self.X, rank1_sol.Xhat).sum(axis=0) ).sum()
        
        if overlap_threshold is not None and self.dict_flow_overlap[1] >= overlap_threshold:
            self.set_optimal_rank(rank=1)
            return

        # other ranks
        for r in range(2, max_rank+1):
            list_solutions = []           
            for _ in range(n_candidates):
                list_solutions.append(
                    ConvexFactoriser.run(X=self.X, rank=r, **kwargs) 
                )

            self.dict_solutions[r] = copy.deepcopy(
                list_solutions[ np.argmin([sol.loss for sol in list_solutions]) ]
            )          

            self.dict_flow_overlap[r] = ( (self.A.sum(axis=0) / self.A.sum()) * np.minimum(self.X, self.dict_solutions[r].Xhat).sum(axis=0) ).sum()
            
            if overlap_threshold is None:
                continue
            if self.dict_flow_overlap[r] >= overlap_threshold:
                self.set_optimal_rank(rank=r)
                return

        return
    
    
    def set_optimal_rank(self, rank):
        """ 
        Set the optimal_rank and optimal_sol variables

        Parameters:
        rank (int)

        Returns:
        None
        """
        self.optimal_rank = rank
        self.optimal_sol = self.dict_solutions[rank]
        self.state_node_userates = (self.A.sum(axis=0)[np.newaxis, :] @ self.optimal_sol.Xhat_in)[0]
        self.state_node_userates /= self.state_node_userates.sum()


    def get_optimal_rank(self, overlap_threshold):
        """ 
        Get the optimal rank given the overlap threshold

        Parameters:
        overlap_threshold (float): threshold of flow overlap, should be < 1.0

        Returns:
        rank = minimum rank such that flow overlap >= overlap_threshold. If no such rank is found, returns max_rank.
        """
        rank = np.argmax(np.array(list(self.dict_flow_overlap.values())) >= overlap_threshold) + 1
        if self.dict_flow_overlap[rank] >= overlap_threshold:
            return rank
        return max(list(self.dict_solutions.keys()))


    def set_rank_1(self):
        """ 
        Set the optimum rank as 1, which has a deterministic solution.

        Parameters:
        None

        Returns:
        None
        """
        self.optimal_rank = 1
        self.optimal_sol = Solution(
            X = self.X, 
            W = np.ones((self.n_preds, 1)), 
            G = np.ones((self.n_preds, 1)), 
            rank = 1, 
        )
    
    def set_rank_M2(self):
        self.optimal_rank = self.n_preds
        self.optimal_sol = Solution(
            X = self.X, 
            W = np.identity(self.n_preds), 
            G = np.identity(self.n_preds), 
            rank = self.n_preds, 
        )
    
    def trim(self, multiplier_ratio=0.01):
        threshold = multiplier_ratio / self.optimal_rank
        
        in_mask = self.optimal_sol.Xhat_in >= threshold
        out_mask = (self.optimal_sol.Xhat_out / self.optimal_sol.Xhat_out.sum(axis=1)[:, np.newaxis]) >= threshold

        self.optimal_sol.Xhat_in = np.where(in_mask, self.optimal_sol.Xhat_in, 0.)
        self.optimal_sol.Xhat_in /= self.optimal_sol.Xhat_in.sum(axis=1)[:, np.newaxis]

        self.optimal_sol.Xhat_out = np.where(out_mask, self.optimal_sol.Xhat_out, 0.)
        self.optimal_sol.Xhat_out /= self.optimal_sol.Xhat_out.sum(axis=0)

        self.optimal_sol.Xhat = self.optimal_sol.Xhat_out @ self.optimal_sol.Xhat_in.T   

