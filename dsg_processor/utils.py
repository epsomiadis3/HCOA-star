class EdgeInfo:
    def __init__(self, adjacency_list, indexes, weights, semantics):
        self.adjacency_list = adjacency_list
        self.indexes = indexes
        self.weights = weights
        self.semantics = semantics

    def __repr__(self):
        return (
            f"EdgeInfo("
            f"adjacency_list={self.adjacency_list}, "
            f"indexes={self.indexes}, "
            f"weights={self.weights}, "
            f"semantics={self.semantics})"
        )
    

class NodeInfo:
    def __init__(self, index_map, coordinates, semantics, parents):
        self.index_map = index_map
        self.coordinates = coordinates
        self.semantics = semantics
        self.parents = parents

    def __repr__(self):
        return (
            f"NodeInfo("
            f"index_map={self.index_map}, "
            f"coordinates={self.coordinates}, "
            f"semantics={self.semantics}, "
            f"parents={self.parents})"
        )
    
    
class PartitionInfo:
    def __init__(self, edge, edge_weights, node_semantics, node_features):
        self.edge = edge
        self.edge_weights = edge_weights
        self.node_semantics = node_semantics
        self.node_features = node_features

    def __repr__(self):
        return (
            f"PartitionInfo("
            f"edge={self.edge}, "
            f"edge_weights={self.edge_weights}, "
            f"node_semantics={self.node_semantics}, "
            f"node_features={self.node_features})"
        )
    