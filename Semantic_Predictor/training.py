import sys
import yaml
from torch_geometric.loader import DataLoader
from torch_geometric.seed import seed_everything
from torch.utils.data import random_split, ConcatDataset
import numpy as np
import copy
import matplotlib.pyplot as plt
import time
import pathlib
import csv
import pickle
import random
from torch_geometric.data import InMemoryDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Define paths
ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
PROJECT_ROOT_PATH = ROOT_PATH.parent.parent
sys.path.append(PROJECT_ROOT_PATH)


# Create Dataset
class GDataset(InMemoryDataset):
    def __init__(self, root, rooms, graph_paths):
        self.graph_paths = graph_paths 
        self.data_list = []
        
        # Load data for each room
        for room in rooms:
            graph_path = graph_paths / f"graphs_room{room}.pt"
            room_data = torch.load(graph_path, weights_only=False)
            self.data_list.extend(room_data)  # Combine the data from each room
        
        super().__init__(root)
        self.process()
        self.load(self.processed_paths[0])  # Load the combined data from the processed path

    @property
    def processed_file_names(self):
        return 'data.pt'
    
    def process(self):
        self.save(self.data_list, self.processed_paths[0])  # Save the combined data
    
    def __repr__(self):
        return f"GDataset with {len(self.data_list) if self.data_list else 0} graphs"


# ================================================================ MC ================================================================
# MC: Get most frequent node label per graph
def get_most_frequent_labels(batch):
    all_most_frequent_labels = []
    all_true_labels = []

    for graph_idx in batch.batch.unique():
        mask = batch.batch == graph_idx # Create mask for nodes in the current graph
        graph_x = batch.x[mask] # Extract node features for this graph
        label_one_hot = graph_x[:, 1:]  # Skip first column, since it gives the border nodes
        labels = label_one_hot.argmax(dim=1).tolist()  # Convert one-hot to class indices
        most_frequent_label = Counter(labels).most_common(1)[0][0]  # Get most frequent class
        graph_y = batch.y[graph_idx]
        all_true_labels.append(graph_y)
        all_most_frequent_labels.append(most_frequent_label)
        
    return torch.tensor(all_most_frequent_labels, dtype=torch.long), torch.tensor(all_true_labels, dtype=torch.long)


# Evaluate the MC model
def evaluate_mc(loader, device):
    y_true, y_pred = [], []
    
    for batch in loader:
        batch = batch.to(device)

        # Predicted graph label: most frequent node label in the graph
        most_frequent_labels, true_labels = get_most_frequent_labels(batch) 
        y_pred.append(most_frequent_labels.detach().cpu())
        y_true.append(true_labels.detach().cpu())

    # Convert to numpy arrays for evaluation
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    return np.mean(y_true == y_pred)


# ================================================================ k-NN ================================================================
def extract_graph_features(data):
    """
    Extract graph-level features from a PyTorch Geometric data object.

    Features:
    1-3. Proportions of semantic classes
    4. Majority class index among border nodes
    """
    features = []

    num_nodes = data.x.size(0)

    # Semantic class one-hot encoding is in columns 1, 2, 3
    semantic = data.x[:, 1:]  # Shape: [num_nodes, num_classes=3]

    class_counts = semantic.sum(dim=0)

    proportions = (class_counts / num_nodes).tolist()
    features.extend(proportions)  # adds 3 values

    # Border node analysis
    border_mask = data.x[:, 0] > 0  # Border indicator
    border_semantic = semantic[border_mask]

    if border_semantic.size(0) > 0:
        border_class_counts = border_semantic.sum(dim=0)
        max_border_class = torch.argmax(border_class_counts).item()
        features.append(max_border_class)
    else:
        # No border nodes present
        features.extend([-1])

    # Extra Features

    # Average node degree (undirected graph: each edge counted twice)
    # edge_index = data.edge_index
    # degrees = torch.bincount(edge_index[0], minlength=num_nodes)
    # avg_degree = degrees.float().mean().item()
    # features.append(avg_degree)

    # Majority class index (most frequent class overall)
    # majority_class = torch.argmax(class_counts).item()
    # features.append(majority_class)

    return np.array(features)


def extract_features_from_loader(loader, device):
    """Extract features and labels from a DataLoader"""
    all_features = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Extracting features"):
        batch = batch.to(device)
        
        # Process each graph in the batch separately
        for i in range(batch.num_graphs):
            # Get individual graph data
            mask = batch.batch == i
            graph_data = type(batch)() # Initialization
            graph_data.x = batch.x[mask]
            graph_data.edge_index = batch.edge_index[:, torch.isin(batch.edge_index[0], torch.where(mask)[0]) & 
                                                      torch.isin(batch.edge_index[1], torch.where(mask)[0])]
            
            # Remap edge indices to local indices (they had the global indices from the whole batch)
            node_mapping = torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
            node_mapping[mask] = torch.arange(mask.sum(), device=device)
            graph_data.edge_index = node_mapping[graph_data.edge_index]
            
            # Extract features
            features = extract_graph_features(graph_data.cpu())
            all_features.append(features)
            all_labels.append(batch.y[i].cpu().numpy())
    
    return np.array(all_features), np.array(all_labels)


class KNNClassifier:
    def __init__(self, k = 3):
        self.k = k
        self.scaler = StandardScaler() # normalize features
        self.knn = KNeighborsClassifier(n_neighbors=k)
        
    def fit(self, train_loader, device):
        """Fit the k-NN classifier on training data"""
        print("Extracting features for k-NN training...")
        X_train, y_train = extract_features_from_loader(train_loader, device)
        
        X_train_scaled = self.scaler.fit_transform(X_train) # Standardize features
        self.knn.fit(X_train_scaled, y_train)
        
        return self
    
    def evaluate(self, test_loader, device):
        """Evaluate k-NN classifier on test data"""
        print("Extracting features for k-NN evaluation...")
        X_test, y_test = extract_features_from_loader(test_loader, device)
        
        X_test_scaled = self.scaler.transform(X_test) # Standardize features
        y_pred = self.knn.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    def predict_single(self, parent, partition_info):
        """Predict using extracted graph features (for inference)"""
        X_test = extract_graph_features(parent, partition_info) # this is the extract_graph_features function for inference in graph_utils.py
        X_test_scaled = self.scaler.transform(X_test)
        prediction = self.knn.predict(X_test_scaled)[0]  # Return scalar prediction

        return prediction


# ================================================================ GNN ================================================================

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


def train(model, device, data_loader, optimizer, loss_fn):
    # Set the model to training mode
    model.train()
    total_loss = 0
    
    # Iterate over the data loader
    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        # Move batch to the device (GPU or CPU)
        batch = batch.to(device)

        # Zero gradients before the backward pass
        optimizer.zero_grad()

        # Feed data into the model to get predictions
        out = model(batch)

        # Ensure the labels are of the correct type (torch.float32)
        labels = batch.y.long()

        # Compute the loss using the filtered output and labels
        loss = loss_fn(out, labels)

        # Accumulate the loss
        total_loss += loss.item()

        # Backpropagate the gradients
        loss.backward()

        # Update the model parameters
        optimizer.step()

    # Return the average loss for the epoch
    return total_loss / len(data_loader)


# Computes accuracy and F1-score for multi-class classification.
def evaluator(y_true, y_pred):
    y_pred_labels = torch.argmax(torch.tensor(y_pred), dim=1).cpu().numpy() # Convert logits to class labels

    return {
        "accuracy": accuracy_score(y_true, y_pred_labels),
        "f1_macro": f1_score(y_true, y_pred_labels, average="macro")  # Macro F1-score
    }


# The evaluation function
def eval(model, device, loader, evaluator, save_model_results=False, save_file=None):
    model.eval()
    y_true, y_pred = [], []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.y.view(-1).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    if save_model_results:
        print ("Saving Model Predictions")

        # Create a pandas dataframe with a two columns
        # y_pred | y_true
        data = {}
        data['y_pred'] = y_pred.reshape(-1)
        data['y_true'] = y_true.reshape(-1)

        df = pd.DataFrame(data=data)
        # Save to csv
        df.to_csv('3dsg_graph_' + save_file + '.csv', sep=',', index=False)

    return evaluator(y_true, y_pred)


# ======================================================================================================================================
# ================================================================ MAIN ================================================================
# ======================================================================================================================================
if __name__ == "__main__":
    # Load config
    data = yaml.safe_load(open(ROOT_PATH / 'config.yaml', 'r'))     

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "No GPU available"

    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"GPU: {gpu_name}")
  
    # If you use GPU, the device should be cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    # Load graph
    room_to_load = [4, 6]
    dataset_test1 = GDataset(root = pathlib.Path("HCOAStar/Semantic_Predictor/data/GNN_combined_dataset"), rooms = room_to_load, graph_paths=ROOT_PATH / data["dataset_labeled"])

    room_to_load = [1, 2]
    dataset_test2 = GDataset(root = pathlib.Path("HCOAStar/Semantic_Predictor/data/GNN_combined_dataset"), rooms = room_to_load, graph_paths=ROOT_PATH / data["dataset_labeled"])

    room_to_load = [3, 5]
    dataset_test3 = GDataset(root = pathlib.Path("HCOAStar/Semantic_Predictor/data/GNN_combined_dataset"), rooms = room_to_load, graph_paths=ROOT_PATH / data["dataset_labeled"])

    room_to_load = [0]
    dataset_test4 = GDataset(root = pathlib.Path("HCOAStar/Semantic_Predictor/data/GNN_combined_dataset"), rooms = room_to_load, graph_paths=ROOT_PATH / data["dataset_labeled"])
    
    # Define datasets
    datasets = [dataset_test1, dataset_test2, dataset_test3, dataset_test4]

    # Split each dataset into train, validation, and test
    train_sets, val_sets, test_sets = [], [], []

    # Set seed for reproducibility
    seed = 0
    generator = torch.Generator().manual_seed(seed)

    for dataset in datasets:
        num_samples = len(dataset)
        train_size = int(0.8 * num_samples)
        val_size = int(0.1 * num_samples)
        test_size = num_samples - train_size - val_size  # Ensures all data is used
        
        train_part, val_part, test_part = random_split(dataset, [train_size, val_size, test_size], generator=generator)
        train_sets.append(train_part)
        val_sets.append(val_part)
        test_sets.append(test_part)  # Keep test sets separate

    # Merge training and validation datasets
    train_dataset = ConcatDataset(train_sets)
    valid_dataset = ConcatDataset(val_sets)

    # Split dataset
    total_len = len(train_dataset)
    half_len = total_len // 2
    train_half_dataset, _ = random_split(train_dataset, [half_len, total_len - half_len], generator=generator)

    # Create new DataLoader with half the dataset
    train_half_loader = DataLoader(train_half_dataset, batch_size=64, shuffle=True, num_workers=0)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Keep separate test DataLoaders for analysis
    test_loaders = [DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0) for test_set in test_sets]

    # =============================================== k-NN ==========================================================
    print("=" * 70)
    print("TRAINING k-NN CLASSIFIER")
    print("=" * 70)
    
    start_time_kNN = time.time()

    k_param = 5
    knn_classifier = KNNClassifier(k = k_param)
    knn_classifier.fit(train_half_loader, device)

    end_time_kNN = time.time()
    total_duration_kNN = end_time_kNN - start_time_kNN


    # =============================================== GNN ==========================================================
    print("=" * 70)
    print("TRAINING GNN CLASSIFIER")
    print("=" * 70)

    # Model parameters 
    num_layers = 3
    input_dim = 4  # Node feature dimension
    hidden_dim = 32
    output_dim = 3  # Number of classes
    dropout = 0.2
    lr =  0.01
    epochs = 5#1600

    # Model, loss, optimizer
    model = GCN_Graph(input_dim, hidden_dim, output_dim, num_layers, dropout).to(device)
    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    loss_fn = nn.CrossEntropyLoss()

    best_model = None
    best_valid_acc = 0
    best_epoch = 0
    loss_history = []

    start_time = time.time()

    for epoch in range(1, 1 + epochs):
        epoch_start_time = time.time()

        print('Training...')
        loss = train(model, device, train_loader, optimizer, loss_fn)
        loss_history.append(loss)

        print('Evaluating...')
        train_result = eval(model, device, train_loader, evaluator)
        val_result = eval(model, device, valid_loader, evaluator)

        train_acc, valid_acc = train_result["accuracy"], val_result["accuracy"]
        
        # Evaluate on all test datasets separately
        test_accuracies = []
        for i, test_loader in enumerate(test_loaders):
            test_result = eval(model, device, test_loader, evaluator)
            test_accuracies.append(test_result["accuracy"])

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = copy.deepcopy(model)
            best_epoch = epoch
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time  # Time per epoch

        # Print results
        print(f'Epoch: {epoch:02d}, '
            f'Loss: {loss:.4f}, '
            f'Train: {100 * train_acc:.2f}%, '
            f'Valid: {100 * valid_acc:.2f}%, '
            f'Time: {epoch_duration:.2f} sec')

        for i, test_acc in enumerate(test_accuracies):
            bn = ['1-20', '21-30', '31-40', '41-50'][i]  
            print(f'Test ({bn} bn): {100 * test_acc:.2f}%')

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Total training time: {total_duration:.2f} seconds ({total_duration / 60:.2f} minutes)")

    ##############################################################
    # COMPARISON #################################################
    ##############################################################
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    # MC
    # Compute accuracy
    MC_train_acc = evaluate_mc(train_loader, device)
    MC_valid_acc = evaluate_mc(valid_loader, device)
    MC_test_accs = []
    for i, test_loader in enumerate(test_loaders):
        test_result = evaluate_mc(test_loader, device)
        MC_test_accs.append(test_result)

    # Print results in the desired format
    print(f'MC: '
        f'Train: {100 * MC_train_acc:.2f}%, '
        f'Valid: {100 * MC_valid_acc:.2f}%')

    for i, test_acc in enumerate(MC_test_accs):
        bn = ['1-20', '21-30', '31-40', '41-50'][i]  
        print(f'Test ({bn} bn): {100 * test_acc:.2f}%')


    # k-NN
    print("=" * 70)
    KNN_train_acc = knn_classifier.evaluate(train_loader, device)
    KNN_valid_acc = knn_classifier.evaluate(valid_loader, device)
    KNN_test_accs = []
    for i, test_loader in enumerate(test_loaders):
        test_result = knn_classifier.evaluate(test_loader, device)
        KNN_test_accs.append(test_result)

    # Print results in the desired format
    print(f'k-NN: '
        f'Train: {100 * KNN_train_acc:.2f}%, '
        f'Valid: {100 * KNN_valid_acc:.2f}%')

    for i, test_acc in enumerate(KNN_test_accs):
        bn = ['1-20', '21-30', '31-40', '41-50'][i]  
        print(f'Test ({bn} bn): {100 * test_acc:.2f}%')
    
    print(f"k-NN training completed in {total_duration_kNN:.2f} seconds")

    # BEST GNN MODEL
    print("=" * 70)
    train_result = eval(best_model, device, train_loader, evaluator)
    val_result = eval(best_model, device, valid_loader, evaluator)

    train_acc, valid_acc = train_result["accuracy"], val_result["accuracy"]

    # Evaluate on all test datasets separately
    test_accuracies = []
    for i, test_loader in enumerate(test_loaders):
        test_result = eval(best_model, device, test_loader, evaluator)
        test_accuracies.append(test_result["accuracy"])

    # Print results
    print(f'Best GNN model found at epoch {best_epoch}: '
        f'Train: {100 * train_acc:.2f}%, '
        f'Valid: {100 * valid_acc:.2f}%')

    for i, test_acc in enumerate(test_accuracies):
        bn = ['1-20', '21-30', '31-40', '41-50'][i]  
        print(f'Test ({bn} bn): {100 * test_acc:.2f}%')

    # =============================================== Save Best Model, Data, and Plots ==========================================================

    save_knn_prompt = input("Do you want to save the k-NN model? (yes/no): ").strip().lower()
    if save_knn_prompt in ['yes', 'y']:
        knn_model_path = ROOT_PATH / "Semantic_Predictor" / "knn_model.pkl"
        knn_model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(knn_model_path, 'wb') as f:
            pickle.dump(knn_classifier, f)
        print(f"k-NN model saved to {knn_model_path}")
    else:
        print("k-NN model was not saved.")

    window_size = 5  # smoothing parameter
    # Compute the moving average
    smoothed_loss = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
    smoothed_epochs = range(window_size, epochs + 1)  # Adjust epoch range to match smoothed values

    save_prompt = input("Do you want to save the model and loss history? (yes/no): ").strip().lower()
    if save_prompt in ['yes', 'y']:
        model_path = ROOT_PATH / "Semantic_Predictor/best_gnn_model.pth"
        torch.save(best_model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        loss_path = ROOT_PATH / "Semantic_Predictor" / "data" / "loss_history.csv"
        with open(loss_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Loss", "Smoothed_Loss"])

            for i in range(epochs):
                smooth_value = smoothed_loss[i - window_size + 1] if i >= window_size - 1 else None
                writer.writerow([i + 1, loss_history[i], smooth_value])
        
        print(f"Loss history saved to {loss_path}")
    else:
        print("Model and loss history were not saved.")

    plt.figure(figsize=(8, 5))
    # Plot original loss
    plt.plot(range(1, epochs + 1), loss_history, linestyle='-', color='b', alpha=0.2, label="Training Loss")

    # Plot the smoothed loss curve in the same color
    plt.plot(smoothed_epochs, smoothed_loss, linestyle='-', color='b', linewidth=2, label="Smoothed Training Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.show()