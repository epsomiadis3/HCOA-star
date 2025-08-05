import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def reassign_labels_by_geometry(radius, center, label, coordinates_list, label_list):
    """
    Reassigns labels to nodes based on geometric location relative to a cylindrical region.
    """
    for idx, place_coords in enumerate(coordinates_list):
        x, y, z = place_coords

        # Check if the place is within the cylinder
        if np.linalg.norm(np.array([x, y]) - np.array(center[:2])) <= radius:
            label_list[idx] = label

    return label_list


# It is used for generating the dataset for the GNN training
def assign_random_labels(place_label_partition, place_coordinates_partition):
    """
    Randomly assigns labels to subgraphs using geometric rules.
    """
    # Initialization
    num = 5  # Number of randomly selected nodes per partition

    for i in range(len(place_coordinates_partition)):
        selected_nodes = np.random.choice(len(place_coordinates_partition[i]), size=num, replace=False)  
        centers = [place_coordinates_partition[i][node] for node in selected_nodes]  # Get corresponding center coordinates
        radii = np.random.uniform(1, 3, size=num)  # Pick random radii between 1 and 6
        labels = np.random.choice([0, 1], size=num)  # Pick random classes (0 or 1)

        # Reassign labels based on geometry
        for k in range(num):
            place_label_partition[i] = reassign_labels_by_geometry(radii[k], centers[k], labels[k], place_coordinates_partition[i], place_label_partition[i])

    return place_label_partition

# ======================================================================================================================================
# ==============================================           MC CODE           ==========================================================
# ======================================================================================================================================

def predict_semantics_MC(semantics):
    counter = Counter(semantics)
    return counter.most_common(1)[0][0]  # Returns the most frequent integer


# ======================================================================================================================================
# ==============================================           kNN CODE           ==========================================================
# ======================================================================================================================================


def extract_graph_features(parent, partition_info):
    """
    Extract graph-level features:
    1-3. Proportions of semantic classes
    4. Majority class index among border nodes
    """
    features = []

    node_labels = partition_info.node_semantics[parent]  # [num_nodes]
    border_nodes = partition_info.node_features[parent]  # [num_nodes]
    num_nodes = len(border_nodes)

    counter = Counter(node_labels)
    for cls in range(3):
        features.append(counter[cls] / num_nodes if cls in counter else 0.0)

    # Border node analysis
    border_semantic = []
    for i in range(num_nodes):
        if border_nodes[i]:
            border_semantic.append(node_labels[i])

    if len(border_semantic) > 0:
        border_counter = Counter(border_semantic)
        majority_class_border = border_counter.most_common(1)[0][0]
        features.append(majority_class_border)
    else:
        # No border nodes present
        features.extend([-1])
    
    result = np.array(features).reshape(1, -1)  # Reshape to 2D for sklearn
    
    return result


class KNNClassifier:
    def __init__(self, k = 5):
        self.k = k
        self.scaler = StandardScaler() # normalize features
        self.knn = KNeighborsClassifier(n_neighbors=k)
    
    def predict_single(self, parent, partition_info):
        """Predict using extracted graph features (for inference)"""
        X_test = extract_graph_features(parent, partition_info)
        X_test_scaled = self.scaler.transform(X_test)
        prediction = self.knn.predict(X_test_scaled)[0]  # Return scalar prediction

        return prediction


def predict_semantics_kNN(parent, partition_info, model = None):

    predicted_semantic_class = model.predict_single(parent, partition_info)
    
    return predicted_semantic_class
    

def kNN_load_model(root):

    with open(root / "Semantic_Predictor" / "knn_model.pkl", 'rb') as f:
        model = pickle.load(f)

    return model

# ======================================================================================================================================
# ==============================================           GNN CODE           ==========================================================
# ======================================================================================================================================


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.0):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, return_embeds=False):
        super(GCN, self).__init__()

        self.pre_mlp = MLP(input_dim, hidden_dim, hidden_dim)  # Preprocessing MLP

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(hidden_dim, hidden_dim))  # First GCN layer
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))  # Hidden layers
        self.convs.append(GCNConv(hidden_dim, output_dim))  # Output layer

        self.bns = nn.ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers)])

        self.dropout = dropout
        self.return_embeds = return_embeds

    def reset_parameters(self):
        self.pre_mlp.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.pre_mlp(x)  # Apply preprocessing MLP

        skip_x = x  # Save for skip connection

        for i in range(len(self.bns)):
            x = self.convs[i](x, edge_index, edge_weight)  # Apply GCN layer
            x = self.bns[i](x)  # Apply BatchNorm
            x = F.relu(x)  # Apply ReLU activation

            # Skip connection (element-wise sum)
            x = x + skip_x
            skip_x = x  # Update skip connection for next layer

            x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout

        x = self.convs[-1](x, edge_index, edge_weight)  # Last layer (no activation)

        return x


class GCN_Graph(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_Graph, self).__init__()

        self.gnn_node = GCN(input_dim, hidden_dim, hidden_dim, num_layers, dropout, return_embeds=True)
        self.pool = global_mean_pool
        self.post_mlp = MLP(hidden_dim, hidden_dim, output_dim)  # Post-processing MLP

    def reset_parameters(self):
        self.gnn_node.reset_parameters()
        self.post_mlp.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

    def forward(self, batched_data):
        x, edge_index, edge_weight, batch = batched_data.x, batched_data.edge_index, batched_data.edge_weight, batched_data.batch

        node_embeddings = self.gnn_node(x, edge_index, edge_weight)
        graph_embeddings = self.pool(node_embeddings, batch)
        out = self.post_mlp(graph_embeddings)  # Apply post-processing MLP

        return out


def predict_semantics_GNN(parent, partition_info, num_classes, device = 'cpu', model = None):

    # Prepare the graph data
    edges = torch.tensor(partition_info.edge[parent], dtype=torch.long).squeeze(1).T  # [2, num_edges]
    edges_rev = edges.flip(0)
    edges_combined = torch.cat([edges, edges_rev], dim=1)
    edge_weights = torch.tensor(partition_info.edge_weights[parent], dtype=torch.float)  # [num_edges]
    edge_weights_combined = torch.cat([edge_weights, edge_weights], dim=0)  # Duplicate weights

    node_features = torch.tensor(partition_info.node_features[parent], dtype=torch.float).unsqueeze(1)  # [num_nodes, num_features]
    node_labels = torch.tensor(partition_info.node_semantics[parent], dtype=torch.long)  # [num_nodes]
    node_labels_onehot = F.one_hot(node_labels, num_classes=num_classes).float()  # One-hot encode node labels
    new_node_features = torch.cat([node_features, node_labels_onehot], dim=1)  # Concatenate features and labels

    # Move graph data to the same device as the model
    new_node_features = new_node_features.to(device)
    edges_combined = edges_combined.to(device)
    edge_weights_combined = edge_weights_combined.to(device)

    # Create the PyG Data object
    graph = Data(
        x=new_node_features,       # Node features
        edge_index=edges_combined,          # Edge indices
        edge_weight=edge_weights_combined,  # Edge weights (optional)
    )

    # Perform inference (model output will be logits)
    with torch.no_grad():  # Disable gradient tracking for inference
        logits = model(graph)  # [num_nodes, num_classes]

    # Get the predicted semantic class for each node (index with highest logit)
    predicted_semantic_class = torch.argmax(logits, dim=1)  # [num_nodes], gives class index

    return predicted_semantic_class


def GNN_load_model(device, num_classes, root):

    num_layers = 3
    input_dim = 4  # Node feature dimension
    hidden_dim = 32
    output_dim = num_classes  # Output dimension should match number of classes
    dropout = 0.2
    model = GCN_Graph(input_dim, hidden_dim, output_dim, num_layers, dropout).to(device)

    # Load the trained model state dictionary
    model.load_state_dict(torch.load(root / "Semantic_Predictor/best_gnn_model.pth", weights_only=False))
    model.eval()

    return model