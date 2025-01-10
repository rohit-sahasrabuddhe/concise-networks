import numpy as np, pandas as pd
from joblib import Parallel, delayed
from .PhysicalNode import PhysicalNode
from .StateNode import StateNode
import copy


class Network():
    """
    Contains properties and methods to process everything at a network level.
    """
    def __init__(self):
        pass

    @classmethod
    def from_paths(cls, paths, flag_n_paths=False):
        """
        Constructor. Creates a Network object with paths. The following variables are created and populated:
        df_trigrams: pd.DataFrame with columns ['i', 'j', 'k', 'num]
        arr_physical_nodes: np.array of physical node ids
        arr_trigram_nodes: np.array of the node ids of physical nodes which have trigrams through them
        dict_physical_nodes: dict of {node id: PhysicalNode}

        
        Parameters:
        paths (list of lists): Each list within is a path.
        flag_n_paths (boolean, default False): if True, the last element of every path is assumed to be a numeric value counting the number of occurences.

        Returns:
        Network
        """
        net = cls()
        net._load_paths(paths, flag_n_paths)
        net._create_physical_nodes()
        return net

    
    def _load_paths(self, paths, flag_n_paths):
        """
        Parses paths into trigrams and populates df_trigrams.
        
        Parameters:
        paths (list of lists): Each list within is a path.
        flag_n_paths (boolean, default False): if True, the last element of every path is assumed to be a numeric value counting the number of occurences.

        Returns:
        None
        """
        
        dict_trigrams = dict()
        for w in paths:
            if flag_n_paths:
                n = float(w[-1])
                for i in range(len(w) - 3):
                    try:
                        dict_trigrams[(w[i], w[i+1], w[i+2])] += n
                    except:
                        dict_trigrams[(w[i], w[i+1], w[i+2])] = n     
            else:
                for i in range(len(w) - 2):
                    try:
                        dict_trigrams[(w[i], w[i+1], w[i+2])] += 1.0
                    except:
                        dict_trigrams[(w[i], w[i+1], w[i+2])] = 1.0       
        
        self.df_trigrams = pd.Series(dict_trigrams)\
            .to_frame('num')\
            .reset_index()\
            .rename(columns={'level_0':'i', 'level_1':'j', 'level_2':'k'})



    def _create_physical_nodes(self):
        """
        Creates physical nodes from the trigrams. The following variables are created:
        arr_physical_nodes: numpy array of physical node ids
        arr_trigram_nodes: numpy array of the node ids of physical nodes which have trigrams through them
        dict_physical_nodes: dict of {node id: PhysicalNode}

        Parameters:
        None

        Returns:
        None
        """

        self.arr_physical_nodes = np.unique(self.df_trigrams[['i', 'j', 'k']].to_numpy())        
        self.arr_trigram_nodes = np.unique(self.df_trigrams['j'].to_numpy())
        arr_has_trigrams = np.isin(self.arr_physical_nodes, self.arr_trigram_nodes, assume_unique=True)

        self.dict_physical_nodes = dict()
        for node_id, has_trigrams in zip(self.arr_physical_nodes, arr_has_trigrams):
            if has_trigrams:
                data = self.df_trigrams[self.df_trigrams['j'] == node_id][['i', 'k', 'num']].copy().reset_index(drop=True)
                arr_preds = np.unique(data['i'].to_numpy())
                arr_succs = np.unique(data['k'].to_numpy())
                A = np.zeros((len(arr_succs), len(arr_preds)))

                temp_pred2idx = dict(zip(arr_preds, range(len(arr_preds))))
                temp_succ2idx = dict(zip(arr_succs, range(len(arr_succs))))

                for row in data.to_numpy():
                    A[temp_succ2idx[row[1]], temp_pred2idx[row[0]]] += row[2]
                
                self.dict_physical_nodes[node_id] = PhysicalNode.from_adjacency_matrix(
                    A=A, id=node_id, arr_preds=arr_preds, arr_succs=arr_succs
                )
            
            else:
                data = self.df_trigrams[self.df_trigrams['i'] == node_id].groupby('j')['num'].sum()
                arr_succs = data.index.to_numpy()
                if len(arr_succs) == 0:
                    M1 = None
                else:
                    M1 = data.to_numpy()
                    M1 /= M1.sum()                
                
                arr_preds = np.unique(self.df_trigrams[self.df_trigrams['k'] == node_id]['j'].to_numpy())
                self.dict_physical_nodes[node_id] = PhysicalNode.from_M1(
                    id=node_id, arr_preds=arr_preds, arr_succs=arr_succs, M1=M1
                )
    


    def process_nodes(self, nodes_to_process='all', flag_parallelise=False, verbose=False, mu=None, all_trigram_nodes=False, **kwargs):
        """
        Process all physical nodes with trigrams. 
        Those that are not processed will be set to rank 1. The others will be given an optimal rank if the overlap_threshold is provided and met.

        Parameters:
        nodes_to_process: 'all' or list of ids of nodes to be factorised. The rest will be set to rank 1.
        flag_parallelise (boolean, default False): use joblib to parallelise
        verbose (boolean, default False)
        mu (None / a numeric value, default None): to be passed on to PhysicalNode.create_adjusted_transition_matrix
        all_trigram_nodes (boolean, default False): create X for all trigram nodes
        **kwargs: for PhysicalNode.process

        Returns:
        None
        """
        def process_j(node_id, verbose,  **kwargs):
            self.dict_physical_nodes[node_id].create_adjusted_transition_matrix(mu=mu) 

            if not node_id in nodes_to_process:
                self.dict_physical_nodes[node_id].set_rank_1()
                if verbose:
                    print(f'Node {node_id} not processed | rank = 1')
                return
            
            if verbose:
                print(f'Processing node {node_id}')
            self.dict_physical_nodes[node_id].process(**kwargs)
            return

        def process_j_parallel(node_id, **kwargs):
            node = copy.deepcopy(self.dict_physical_nodes[node_id])
            node.create_adjusted_transition_matrix(mu=mu)
            if not node_id in nodes_to_process:
                node.set_rank_1()
                return node
            node.process(**kwargs)
            return node


        if type(nodes_to_process)==str and nodes_to_process == 'all':
            nodes_to_process = list(self.dict_physical_nodes.keys())  
        

        if flag_parallelise:
            if all_trigram_nodes:
                processed_nodes = Parallel(n_jobs=-1)(delayed(process_j_parallel)(node_id, **kwargs) for node_id in self.arr_trigram_nodes)
                for node_id, node in zip(self.arr_trigram_nodes, processed_nodes):
                    self.dict_physical_nodes[node_id] = copy.deepcopy(node)
            else:
                processed_nodes = Parallel(n_jobs=-1)(delayed(process_j_parallel)(node_id, **kwargs) for node_id in nodes_to_process)
                for node_id, node in zip(nodes_to_process, processed_nodes):
                    self.dict_physical_nodes[node_id] = copy.deepcopy(node)
        else:
            if all_trigram_nodes:
                _ = [process_j(node_id, verbose, **kwargs) for node_id in self.arr_trigram_nodes]
            else:
                _ = [process_j(node_id, verbose, **kwargs) for node_id in nodes_to_process]

        return



    def stitch(self):
        """ 
        Stitch the network together from the processed physical nodes. The following variables are created:
        dict_state_nodes: dict of {state node id: StateNode}
        edges: dict of dicts {pred state node id: {succ state node id: weight}}

        Parameters:
        None

        Returns:
        None
        """

        dict_n_state_nodes = dict()

        self.dict_state_nodes = dict()
        self.edges = dict()

        
        # Creating the State Nodes for all physical nodes
        for pn_id, physical_node in self.dict_physical_nodes.items():
            if not physical_node.has_trigrams:
                sn_id = f'{pn_id}-1'
                self.dict_state_nodes[ sn_id ] = StateNode(id=sn_id, physical_node_id=pn_id)
                dict_n_state_nodes[pn_id] = 1
                continue
            
            dict_n_state_nodes[pn_id] = physical_node.optimal_rank
            for r in range(1, physical_node.optimal_rank+1):
                sn_id = f'{pn_id}-{r}'
                self.dict_state_nodes[sn_id] = StateNode(id=sn_id, physical_node_id=pn_id)


        
        for pred_id, pred_pn in self.dict_physical_nodes.items():
            if pred_pn.has_trigrams: # predecessor has trigrams
                pred_rank = dict_n_state_nodes[pred_id]                
                for succ_idx, succ_id in enumerate(pred_pn.arr_succs): # for every successor
                    succ_pn = self.dict_physical_nodes[succ_id]
                    succ_rank = dict_n_state_nodes[succ_id]
                    if pred_id in list(succ_pn.arr_preds): #predecessor is in the successor's pred list
                        pred_idx = list(succ_pn.arr_preds).index(pred_id)
                        if succ_rank > 1:
                            for pred_r in range(1, pred_rank+1):
                                pred_sn_id = f'{pred_id}-{pred_r}'
                                if not (pred_sn_id in self.edges):
                                    self.edges[pred_sn_id] = dict()
                                for succ_r in range(1, succ_rank+1):
                                    succ_sn_id = f'{succ_id}-{succ_r}'
                                    w = pred_pn.optimal_sol.Xhat_out[succ_idx, pred_r-1] * succ_pn.optimal_sol.Xhat_in[pred_idx, succ_r-1]
                                    if w > 0:
                                        self.edges[pred_sn_id][succ_sn_id] = w
                        else:
                            succ_sn_id = f'{succ_id}-1'
                            for pred_r in range(1, pred_rank+1):
                                pred_sn_id = f'{pred_id}-{pred_r}'
                                if not (pred_sn_id in self.edges):
                                    self.edges[pred_sn_id] = dict()
                                w = pred_pn.optimal_sol.Xhat_out[succ_idx, pred_r-1]
                                if w > 0:
                                    self.edges[pred_sn_id][succ_sn_id] = w
                    
                    else: #predecessor is not in the successor's pred list
                        if succ_rank > 1:
                            for pred_r in range(1, pred_rank+1):
                                pred_sn_id = f'{pred_id}-{pred_r}'
                                if not (pred_sn_id in self.edges):
                                    self.edges[pred_sn_id] = dict()
                                for succ_r in range(1, succ_rank+1):
                                    succ_sn_id = f'{succ_id}-{succ_r}'
                                    w = pred_pn.optimal_sol.Xhat_out[succ_idx, pred_r-1] * succ_pn.state_node_userates[succ_r-1]
                                    if w > 0:
                                        self.edges[pred_sn_id][succ_sn_id] = w
                        else:
                            succ_sn_id = f'{succ_id}-1'
                            for pred_r in range(1, pred_rank+1):
                                pred_sn_id = f'{pred_id}-{pred_r}'
                                if not (pred_sn_id in self.edges):
                                    self.edges[pred_sn_id] = dict()
                                w = pred_pn.optimal_sol.Xhat_out[succ_idx, pred_r-1]
                                if w > 0:
                                    self.edges[pred_sn_id][succ_sn_id] = w


            else: # pred does not have trigrams
                if len(pred_pn.arr_succs) != 0: # there are successors
                    pred_sn_id = f'{pred_id}-1'
                    if not (pred_sn_id in self.edges):
                        self.edges[pred_sn_id] = dict()
                    for succ_idx, succ_id in enumerate(pred_pn.arr_succs): # for every successor
                        succ_pn = self.dict_physical_nodes[succ_id]
                        succ_rank = dict_n_state_nodes[succ_id]
                        if pred_id in list(succ_pn.arr_preds): #predecessor is in the successor's pred list
                            pred_idx = list(succ_pn.arr_preds).index(pred_id)
                            if succ_rank > 1:
                                for succ_r in range(1, succ_rank+1):
                                    succ_sn_id = f'{succ_id}-{succ_r}'
                                    w =  pred_pn.M1[succ_idx] * succ_pn.optimal_sol.Xhat_in[pred_idx, succ_r-1]
                                    if w > 0:
                                        self.edges[pred_sn_id][succ_sn_id] =  pred_pn.M1[succ_idx] * succ_pn.optimal_sol.Xhat_in[pred_idx, succ_r-1]
                            else:
                                succ_sn_id = f'{succ_id}-1'
                                w =  pred_pn.M1[succ_idx]
                                if w>0:
                                    self.edges[pred_sn_id][succ_sn_id] = w
                        else: #predecessor is not in the successor's pred list
                            if succ_rank > 1:
                                for succ_r in range(1, succ_rank+1):
                                    succ_sn_id = f'{succ_id}-{succ_r}'
                                    w =  pred_pn.M1[succ_idx] * succ_pn.state_node_userates[succ_r-1]
                                    if w > 0:
                                        self.edges[pred_sn_id][succ_sn_id] = w
                            else:
                                succ_sn_id = f'{succ_id}-1'
                                w = pred_pn.M1[succ_idx]
                                if w > 0:
                                    self.edges[pred_sn_id][succ_sn_id] = w

        return       
