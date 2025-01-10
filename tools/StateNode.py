class StateNode():
    def __init__(self, id, physical_node_id):
        self.id = id
        self.physical_node_id = physical_node_id
        self.in_edges = dict()
        self.out_edges = dict()
        