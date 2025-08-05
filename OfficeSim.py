import sys
import yaml
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import time
import pathlib
import torch
import random
from HCOAStar.dsg_processor.dsg_loader import convert_dsg_office
from HCOAStar.Path_Planner.planning import dijkstra, a_star, class_ordered_a_star, hierarchical_planner, modified_a_star
from HCOAStar.dsg_processor.visualization_utils import plot_dsg_3D, plot_dsg_2D, plot_dsg_2D_total
from HCOAStar.dsg_processor.utils import EdgeInfo, NodeInfo
from HCOAStar.dsg_processor.graph_utils import KNNClassifier, GNN_load_model, kNN_load_model

# Define paths
ROOT_PATH = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT_PATH = ROOT_PATH.parent
sys.path.append(str(PROJECT_ROOT_PATH))

print(sys.version)

if __name__ == "__main__":
    # Load config
    data = yaml.safe_load(open(ROOT_PATH / 'config.yaml', 'r'))   

    # Convert DSG file to graph info and save to graph_info_path
    convert_dsg_office(path_to_dsg = ROOT_PATH / data["dsg_json_path"], output_file = ROOT_PATH / data["graph_info_path"])

    # Load graph info
    f = open(ROOT_PATH / data["graph_info_path"], 'rb')
    place_nodes, place_edges, room_nodes, room_edges, interlayer_edge_list, place_partition_info = pkl.load(f)

    # Data
    start_node = 599 
    end_node = 588 
    n_classes = max(place_edges.semantics) + 1
    semantic_color_list = [
      np.array([1.0, 0.0, 0.0]),  # Red
      np.array([0.0, 0.0, 1.0]),  # Blue
      np.array([0.0, 0.8, 0.0])   # Green (default label)
    ]
    
    print(f"Number of nodes in Room Layer: {len(room_nodes.coordinates)}")
    print(f"Number of nodes in Place Layer: {len(place_nodes.coordinates)}")

# ==============================================           PATH-PLANNING           ==========================================================

    # # Compute shortest path (no semantics)
    # path_dijkstra = dijkstra(place_edges.adjacency_list, start_node, end_node)
    # path_a_star = a_star(place_edges.adjacency_list, start_node, end_node, place_nodes.coordinates)

    # Number of iterations
    N_RUNS = 1

    # Lists to store execution times
    times_COAStar = []
    times_HCOAStar_mc = []
    times_HCOAStar_knn = []
    times_HCOAStar_gnn = []

    # Measure time for COA*
    for _ in range(N_RUNS):
        start_time = time.time()
        path_COAStar, expanded_places_COAStar = class_ordered_a_star(
            place_edges.adjacency_list, start_node, end_node, place_nodes.coordinates, n_classes
        )
        end_time = time.time()
        times_COAStar.append(end_time - start_time)
    print(f"class_ordered_a_star execution time: Mean = {np.mean(times_COAStar):.4f}s, Std = {np.std(times_COAStar):.4f}s")

    # Measure time for HCOA* with MC
    for _ in range(N_RUNS):
        start_time = time.time()
        room_path_HCOAStar_mc, place_path_HCOAStar_mc, expanded_rooms_HCOAStar_mc, expanded_places_HCOAStar_mc = hierarchical_planner(
            place_nodes, place_edges, room_edges.adjacency_list, start_node, end_node, room_nodes.coordinates,
            n_classes, place_partition_info, Sem_Pred = 'MC'
        )
        end_time = time.time()
        times_HCOAStar_mc.append(end_time - start_time)
    print(f"hierarchical_planner (MC) execution time: Mean = {np.mean(times_HCOAStar_mc):.4f}s, Std = {np.std(times_HCOAStar_mc):.4f}s")

    # Measure time for HCOA* with kNN
    kNN_model = kNN_load_model(ROOT_PATH)

    for _ in range(N_RUNS):
        start_time = time.time()
        room_path_HCOAStar_knn, place_path_HCOAStar_knn, expanded_rooms_HCOAStar_knn, expanded_places_HCOAStar_knn = hierarchical_planner(
            place_nodes, place_edges, room_edges.adjacency_list, start_node, end_node, room_nodes.coordinates,
            n_classes, place_partition_info, root=ROOT_PATH, Sem_Pred = 'kNN', model = kNN_model
        )
        end_time = time.time()
        times_HCOAStar_knn.append(end_time - start_time)
    print(f"hierarchical_planner (kNN) execution time: Mean = {np.mean(times_HCOAStar_knn):.4f}s, Std = {np.std(times_HCOAStar_knn):.4f}s")

    # Measure time for HCOA* with GNN
    # seems that running in cpu is faster for small graph inference
    device = 'cpu'   #'cuda' if torch.cuda.is_available() else 'cpu'
    GNN_model =  GNN_load_model(device, n_classes, ROOT_PATH)

    for _ in range(N_RUNS):
        start_time = time.time()
        room_path_HCOAStar_gnn, place_path_HCOAStar_gnn, expanded_rooms_HCOAStar_gnn, expanded_places_HCOAStar_gnn = hierarchical_planner(
            place_nodes, place_edges, room_edges.adjacency_list, start_node, end_node, room_nodes.coordinates,
            n_classes, place_partition_info, root=ROOT_PATH, Sem_Pred = 'GNN', device = device, model = GNN_model
        )
        end_time = time.time()
        times_HCOAStar_gnn.append(end_time - start_time)
    print(f"hierarchical_planner (GNN) execution time: Mean = {np.mean(times_HCOAStar_gnn):.4f}s, Std = {np.std(times_HCOAStar_gnn):.4f}s")


    # Print number of expanded nodes
    print(f"HCOA* expanded {sum(expanded_places_HCOAStar_mc) + sum(expanded_rooms_HCOAStar_mc)}")
    print(f"COA* expanded {sum(expanded_places_COAStar)}")
       

# ==============================================           SUBOPTIMALITY           ==========================================================

    def compute_path_match_percentage(reference_path, comparison_path):
        max_len = max(len(reference_path), len(comparison_path))
        if max_len == 0:
            return 0.0
        matches = sum(
            1 for i in range(max_len)
            if i < len(reference_path) and i < len(comparison_path) and reference_path[i] == comparison_path[i]
        )

        return 100.0 * matches / max_len
    
    random.seed(1)
    np.random.seed(1)
    n_runs = 500
    mastar_matches = []
    mc_matches = []
    knn_matches = []
    gnn_matches = []
    num_nodes = len(place_nodes.coordinates)

    for _ in range(n_runs):
        # Random start and end nodes
        start_node = random.randint(0, num_nodes - 1)
        end_node = random.randint(0, num_nodes - 1)
        while end_node == start_node:
            end_node = random.randint(0, num_nodes - 1)

        # COA*
        path_COAStar_Sim2, _ = class_ordered_a_star(
            place_edges.adjacency_list, start_node, end_node, place_nodes.coordinates, n_classes
        )

        # Modified A*
        alpha = 2
        path_MAStar, _ = modified_a_star(
            place_edges.adjacency_list, start_node, end_node, place_nodes.coordinates, n_classes, alpha
        )

        # HCOA* MC
        _, path_mc, _, _ = hierarchical_planner(
            place_nodes, place_edges, room_edges.adjacency_list,
            start_node, end_node, room_nodes.coordinates,
            n_classes, place_partition_info, Sem_Pred='MC'
        )

        # HCOA* kNN
        _, path_knn, _, _ = hierarchical_planner(
            place_nodes, place_edges, room_edges.adjacency_list,
            start_node, end_node, room_nodes.coordinates,
            n_classes, place_partition_info, root=ROOT_PATH,
            Sem_Pred='kNN', model=kNN_model
        )

        # HCOA* GNN
        _, path_gnn, _, _ = hierarchical_planner(
            place_nodes, place_edges, room_edges.adjacency_list,
            start_node, end_node, room_nodes.coordinates,
            n_classes, place_partition_info, root=ROOT_PATH,
            Sem_Pred='GNN', device=device, model=GNN_model
        )

        # Compare with COA*
        mastar_matches.append(compute_path_match_percentage(path_COAStar_Sim2, path_MAStar))
        mc_matches.append(compute_path_match_percentage(path_COAStar_Sim2, path_mc))
        knn_matches.append(compute_path_match_percentage(path_COAStar_Sim2, path_knn))
        gnn_matches.append(compute_path_match_percentage(path_COAStar_Sim2, path_gnn))

    # Report average match percentages
    print(f"Average Match with COA* over {len(mc_matches)} valid runs:")
    print(f"Modified A*:   {np.mean(mastar_matches):.2f}%")
    print(f"MC:   {np.mean(mc_matches):.2f}%")
    print(f"kNN:  {np.mean(knn_matches):.2f}%")
    print(f"GNN:  {np.mean(gnn_matches):.2f}%")


# ==============================================           FIGURES           ==========================================================

    # Plot colored places layer
    plot_dsg_2D(place_nodes.coordinates, place_edges.indexes, place_nodes.semantics, semantic_color_list)

    # Plot COA* path
    plot_dsg_2D(place_nodes.coordinates, place_edges.indexes, place_nodes.semantics, semantic_color_list, path = path_COAStar, expanded_nodes = expanded_places_COAStar)

    # Plot HCOA* room path
    plot_dsg_2D(room_nodes.coordinates, room_edges.indexes, room_nodes.semantics, semantic_color_list, path = room_path_HCOAStar_mc, expanded_nodes = expanded_rooms_HCOAStar_mc)

    # Plot COA* places path
    plot_dsg_2D(place_nodes.coordinates, place_edges.indexes, place_nodes.semantics, semantic_color_list, path = place_path_HCOAStar_mc, expanded_nodes = expanded_places_HCOAStar_mc)

    # Plot 3DSG graph
    plot_dsg_3D(place_nodes.coordinates, place_edges.indexes, room_nodes.coordinates, room_edges.indexes, interlayer_edge_list)