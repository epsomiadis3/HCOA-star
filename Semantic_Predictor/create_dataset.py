try:
    import spark_dsg as dsg
except ImportError:
    print("Spark_DSG not loaded")
import sys
import yaml
import random
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
import pathlib
import pickle as pkl
from HCOAStar.dsg_processor.dsg_loader import extract_nodes, get_node_index
from HCOAStar.dsg_processor.graph_utils import assign_random_labels
from HCOAStar.Path_Planner.planning import class_ordered_a_star
from HCOAStar.dsg_processor.visualization_utils import plot_dsg_2D_with_features_subgraphs, plot_dsg_2D

# Define paths
ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
PROJECT_ROOT_PATH = ROOT_PATH.parent.parent
sys.path.append(PROJECT_ROOT_PATH)

    
# Processing DSG: Edges
def generate_connectivity(G: dsg.DynamicSceneGraph, place_nodes, room_nodes, place_partition):
    print("Generating connectivity...")

    # Initialization
    place_index_map = place_nodes.index_map
    place_coordinates_list = place_nodes.coordinates
    place_parent_list = place_nodes.parents
    num_places = len(place_coordinates_list)  # Number of place nodes
    num_rooms = len(room_nodes.coordinates)  # Number of room nodes
    edge_partition = [[] for _ in range(num_rooms)]  # List of edges for each place
    edge_weights_partition = [[] for _ in range(num_rooms)]  # Corresponding edge weights
    border_nodes = [0] * num_places  # List for place features
    edges_DSG = G.get_layer(dsg.DsgLayers.PLACES).edges
    
    for edge in edges_DSG:
        start_node, end_node = G.get_node(edge.source), G.get_node(edge.target)

        if start_node.id.value in place_index_map[1] and end_node.id.value in place_index_map[1]:
            start_index = get_node_index(start_node, place_index_map)
            end_index = get_node_index(end_node, place_index_map)

            parent_start = place_parent_list[start_index]
            parent_end = place_parent_list[end_index]

            if parent_start == parent_end:
                # Compute the edge length (Euclidean distance)
                edge_len = np.linalg.norm(place_coordinates_list[start_index] - place_coordinates_list[end_index])

                # Avoid duplicate edges and self-loops
                if (start_index, end_index) not in edge_partition[parent_start] and \
                   (end_index, start_index) not in edge_partition[parent_start] and \
                   start_index != end_index:                
                    # Edge numbering in each partition should start from 0
                    node_length = sum(len(sublist) for sublist in place_partition[:parent_start])
                    modified_edge = (start_index - node_length, end_index - node_length)
                    edge_partition[parent_start].append(modified_edge)
                    edge_weights_partition[parent_start].append(edge_len)
            else:
                border_nodes[start_index] = 1
                border_nodes[end_index] = 1

        else:
            print(f"Minor warning: Edge ({start_node.id.value}, {end_node.id.value}) not in initial layer No. {dsg.DsgLayers.PLACES}, probably because one of the nodes did not have a parent")

                
    print("   Done!")
    return edge_partition, edge_weights_partition, border_nodes


def convert_dsg(path_to_dsg, output_file="./data/graph_partitions.pkl"):
    # Load DSG
    path_to_dsg = pathlib.Path(path_to_dsg).expanduser().absolute()
    G = dsg.DynamicSceneGraph.load(str(path_to_dsg))
    print("DSG loaded")

    # Extract node info and induced subgraphs
    place_nodes, room_nodes, place_partition = extract_nodes(G = G)

    # Generate connectivity
    edge_partition, edge_weights_partition, place_features = generate_connectivity(G = G, place_nodes = place_nodes, room_nodes = room_nodes, place_partition = place_partition)

    place_features_partition = [[place_features[i] for i in sublist] for sublist in place_partition]
    place_coordinates_partition = [[place_nodes.coordinates[i] for i in sublist] for sublist in place_partition]
    
    print("   Done!")
    print("Saving pkl file...")
    f = open(output_file, "wb")
    pkl.dump([place_coordinates_partition, place_partition, edge_partition, edge_weights_partition, place_features_partition], f)
    print("   Done!")


def compute_adj_list(edges, edge_weights, place_labels):
    # Initialization
    n_nodes = len(place_labels)
    adjacency_list = {i: [] for i in range(n_nodes)}

    # Create adjacency list
    for i, (u, v) in enumerate(edges):
        weight = edge_weights[i]
        label = min(place_labels[u], place_labels[v])
        adjacency_list[u].append((v, weight, label))
        adjacency_list[v].append((u, weight, label))

    return adjacency_list


# Main function
if __name__ == "__main__":
    # Load config
    data = yaml.safe_load(open(ROOT_PATH / 'config.yaml', 'r'))   

    # Initialization
    n_graphs = 2000 # number of graphs created from each room
    place_label_partition_all_graphs = []
    paths_all_graphs = []
    labels_all_graphs = []
    n_labels = 3
    label_colors = [
        np.array([1.0, 0.0, 0.0]),  # Red
        np.array([0.0, 0.0, 1.0]),  # Blue
        np.array([0.0, 0.8, 0.0])   # Green (default label)
    ]
    
    # Convert DSG file to graph info and save to graph_partitions
    convert_dsg(path_to_dsg = ROOT_PATH / data["dsg_json_path"], output_file = ROOT_PATH / data["graph_partitions"])

    # Load graph info
    f = open(ROOT_PATH / data["graph_partitions"], 'rb')
    place_coordinates_partition, place_partition, edge_partition, edge_weights_partition, place_features_partition = pkl.load(f)

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Assign different node labels and run COA* in each room from different border nodes
    for i in range(n_graphs):
        paths = []
        graph_label = []
        place_label_partition = [[n_labels - 1] * len(sublist) for sublist in place_partition] # initialization with high priority class
        place_label_partition = assign_random_labels(place_label_partition, place_coordinates_partition)
        place_label_partition_all_graphs.append(place_label_partition)

        # Compute optimal path and least priority label in the path
        for k in range(len(place_coordinates_partition)):
            place_features_k = place_features_partition[k]  # Get feature list for current partition

            # Select a random border node as the start node
            feature_indices = np.where(np.array(place_features_k) == 1)[0]
            start_node = np.random.choice(feature_indices)

            # Select a random node outside of a disk from the start node
            start_coords = place_coordinates_partition[k][start_node]
            candidates = [
                idx for idx, coords in enumerate(place_coordinates_partition[k])
                if np.linalg.norm(np.array(coords[:2]) - start_coords[:2]) > 5
            ]

            if len(candidates) == 0:
                print(f"Warning: No far away node found in subgraph {k} as end node, selecting a random one.")
                possible_nodes = [idx for idx in range(len(place_coordinates_partition[k])) if idx != start_node] # Get all possible nodes except start_node
                end_node = np.random.choice(possible_nodes)
            else:
                end_node = np.random.choice(candidates)

            adjacency_list = compute_adj_list(edge_partition[k], edge_weights_partition[k], place_label_partition[k])

            # Compute path using class-ordered A*
            path_class_ordered_a_star, _ = class_ordered_a_star(adjacency_list, start_node, end_node, place_coordinates_partition[k], n_classes = n_labels)

            room_label = min(place_label_partition[k][i] for i in path_class_ordered_a_star)

            paths.append(path_class_ordered_a_star)
            graph_label.append(room_label)
        
        paths_all_graphs.append(paths)
        labels_all_graphs.append(graph_label)

    print("Saving Dataset...")

    # Iterate over all rooms
    for k in range(len(labels_all_graphs[0])):  # Number of rooms
        graph_list = []  # List to store graphs for the current room

        # Iterate over all graphs
        for i in range(len(place_label_partition_all_graphs)):  
            
            # Convert data to PyTorch tensors
            # Original edges
            edges = torch.tensor(edge_partition[k], dtype=torch.long)  # [num_edges, 2]
            edge_weights = torch.tensor(edge_weights_partition[k], dtype=torch.float)  # [num_edges]

            # Duplicate edges for undirected graph
            edges_rev = edges[:, [1, 0]]  # Reverse direction
            edges_combined = torch.cat([edges, edges_rev], dim=0).T  # [2, 2*num_edges]
            edge_weights_combined = torch.cat([edge_weights, edge_weights], dim=0)  # Duplicate weights
            
            node_features = torch.tensor(place_features_partition[k], dtype=torch.float).unsqueeze(1)  # [num_nodes, num_features]
            node_labels = torch.tensor(place_label_partition_all_graphs[i][k], dtype=torch.long)  # [num_nodes]
            node_labels_onehot = F.one_hot(node_labels, num_classes=n_labels).float()  # One-hot encode node labels

            # Combine original features and one-hot encoded labels
            node_features_aggregated = torch.cat([node_features, node_labels_onehot], dim=1)  

            graph_label = torch.tensor(labels_all_graphs[i][k], dtype=torch.long)  # Graph-level label

            # Create PyG Data object
            graph_data = Data(
                x=node_features_aggregated,       # Node features
                edge_index=edges_combined,          # Edge indices
                edge_weight=edge_weights_combined,  # Edge weights (optional)
                y=graph_label              # Graph label
            )

            # Add the current graph to the list
            graph_list.append(graph_data)

        # Save graphs of the current room to a separate file
        file_path = ROOT_PATH / data["dataset_labeled"] / f"graphs_room{k}.pt"
        torch.save(graph_list, file_path)

        print(f"Dataset for Room {k} saved successfully at {file_path}.")

    # PLOTS
    plot_dsg_2D_with_features_subgraphs(place_coordinates_partition, place_partition, edge_partition, place_features_partition, place_label_partition_all_graphs[0], label_colors)
    plot_dsg_2D(place_coordinates_partition[0], edge_partition[0], place_label_partition_all_graphs[0][0], label_colors, path = paths_all_graphs[0][0])
