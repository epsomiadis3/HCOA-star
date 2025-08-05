import heapq
import math
from HCOAStar.dsg_processor.graph_utils import predict_semantics_GNN, predict_semantics_MC, predict_semantics_kNN

def dijkstra(adjacency_list, start_node, end_node):
    # Initialize distances and previous nodes
    n_nodes = len(adjacency_list)
    distances = {node: float('inf') for node in range(n_nodes)}
    distances[start_node] = 0
    previous_nodes = {node: None for node in range(n_nodes)}

    # Priority queue to select the node with the minimum distance
    priority_queue = [(0, start_node)]  # (distance, node)

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # Stop early if we reach the end node
        if current_node == end_node:
            break

        # If the distance is greater than the recorded shortest, skip this iteration
        if current_distance > distances[current_node]:
            continue

        # Check current_node's neighbors
        for neighbor, weight, semantic_class in adjacency_list[current_node]:
            distance = current_distance + weight

            # Only consider this path if it's better
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    # Backtrack to get the path from start_node to end_node
    path = []
    current = end_node
    while current is not None:
        path.append(current)
        current = previous_nodes[current]

    # Reverse path to get the correct order
    path = path[::-1]

    # Return an empty path if start_node and end_node are disconnected
    if path[0] != start_node:
        print("No feasible path using Dijkstra exists!")
        return []

    return path


def heuristic(node, end_node, coordinates_list):
    # Straight-line distance as the heuristic
    x1, y1, z1 = coordinates_list[node]
    x2, y2, z2 = coordinates_list[end_node]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def a_star(adjacency_list, start_node, end_node, coordinates_list):
    # Initialize distances and previous nodes
    n_nodes = len(adjacency_list)
    g_scores = {node: float('inf') for node in range(n_nodes)}  # Cost from start to node
    g_scores[start_node] = 0
    previous_nodes = {node: None for node in range(n_nodes)}

    # Priority queue: (f_score, node)
    priority_queue = [(0, start_node)]  # Initial f_score is 0

    while priority_queue:
        current_f_score, current_node = heapq.heappop(priority_queue)

        # Stop early if we reach the end node
        if current_node == end_node:
            break

        # Check current_node's neighbors
        for neighbor, weight, semantic_class in adjacency_list[current_node]:
            tentative_g_score = g_scores[current_node] + weight

            # If this path is better, update scores
            if tentative_g_score < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g_score
                previous_nodes[neighbor] = current_node

                # Compute f_score: g_score + heuristic
                f_score = tentative_g_score + heuristic(neighbor, end_node, coordinates_list)
                heapq.heappush(priority_queue, (f_score, neighbor))

    # Backtrack to get the path from start_node to end_node
    path = []
    current = end_node
    while current is not None:
        path.append(current)
        current = previous_nodes[current]

    # Reverse path to get the correct order
    path = path[::-1]

    # Return an empty path if start_node and end_node are disconnected
    if path[0] != start_node:
        print("No feasible path exists!")
        return []

    return path


def modified_a_star(adjacency_list, start_node, end_node, coordinates_list, n_classes, alpha):
    # Initialize distances and previous nodes
    n_nodes = len(adjacency_list)
    g_scores = {node: float('inf') for node in range(n_nodes)}  # Cost from start to node
    g_scores[start_node] = 0
    previous_nodes = {node: None for node in range(n_nodes)}
    expanded_nodes = [0] * n_nodes

    # Priority queue: (f_score, node)
    priority_queue = [(0, start_node)]  # Initial f_score is 0

    while priority_queue:
        current_f_score, current_node = heapq.heappop(priority_queue)

        # Stop early if we reach the end node
        if current_node == end_node:
            break

        expanded_nodes[current_node] = 1 # Mark node as expanded

        # Check current_node's neighbors
        for neighbor, weight, semantic_class in adjacency_list[current_node]:

            modified_weight = weight + alpha**(n_classes - semantic_class)

            tentative_g_score = g_scores[current_node] + modified_weight

            # If this path is better, update scores
            if tentative_g_score < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g_score
                previous_nodes[neighbor] = current_node

                # Compute f_score: g_score + heuristic
                f_score = tentative_g_score + heuristic(neighbor, end_node, coordinates_list)
                heapq.heappush(priority_queue, (f_score, neighbor))

    # Backtrack to get the path from start_node to end_node
    path = []
    current = end_node
    while current is not None:
        path.append(current)
        current = previous_nodes[current]

    # Reverse path to get the correct order
    path = path[::-1]

    # Return an empty path if start_node and end_node are disconnected
    if path[0] != start_node:
        print("No feasible path exists!")
        return []

    return path, expanded_nodes


def class_ordered_a_star(adjacency_list, start_node, end_node, coordinates_list, n_classes, partition_info = None, path_HL = None, nodes = None, root = None, Sem_Pred = 'MC', device = 'cpu', model = None):
    # Initialize distances and previous nodes
    n_nodes = len(adjacency_list)
    semantic_class_nodes_pred = [n_classes-1] * n_nodes
    semantic_class_nodes_pred_flag = [False] * n_nodes
    g_scores = {node: float('inf') for node in range(n_nodes)}  # Cost from start to node
    g_scores[start_node] = 0
    theta_scores = {node: [float('inf')] * n_classes for node in range(n_nodes)}
    theta_scores[start_node] = [0] * n_classes
    previous_nodes = {node: None for node in range(n_nodes)}
    expanded_nodes = [0] * n_nodes

    # Priority queue: (theta_score, f_score, node)
    priority_queue = [(theta_scores[start_node], 0, start_node)]

    while priority_queue:
        current_theta_score, current_g_score, current_node = heapq.heappop(priority_queue)

        # Stop early if we reach the end node
        if current_node == end_node:
            break

        if current_theta_score > theta_scores[current_node]:
            continue  # Skip redundant expansion
        
        expanded_nodes[current_node] = 1 # Mark node as expanded

        # Check current_node's neighbors
        for neighbor, weight, semantic_class in adjacency_list[current_node]:
            if nodes is not None:
                if nodes.parents[neighbor] not in path_HL:
                    continue

            # for rooms
            if partition_info is not None and neighbor is not end_node:
                if semantic_class_nodes_pred_flag[neighbor] is False:
                    semantic_class_nodes_pred_flag[neighbor] = True
                    if Sem_Pred == 'MC':
                        semantic_class_nodes_pred[neighbor] = predict_semantics_MC(partition_info.node_semantics[neighbor])
                    elif Sem_Pred == 'kNN':
                        semantic_class_nodes_pred[neighbor] = predict_semantics_kNN(neighbor, partition_info, model = model)
                    elif Sem_Pred == 'GNN':
                        semantic_class_nodes_pred[neighbor] = predict_semantics_GNN(neighbor, partition_info, n_classes, device = device, model = model)
                    
                    # print(f"Parent node {neighbor} is assigned semantic class {semantic_class_nodes_pred[neighbor]}.")
                        
                semantic_class = min(semantic_class_nodes_pred[current_node], semantic_class_nodes_pred[neighbor])
                # print(current_node)
                

            tentative_g_score = g_scores[current_node] + weight
            tentative_theta_score = theta_scores[current_node].copy()
            tentative_theta_score[semantic_class] += 1

            # If this path is better, update scores (compare theta_scores lexicographically first, that is if first element is smaller there is no need to check the second element, that is why we put the least priority class as first element)
            if tentative_theta_score < theta_scores[neighbor] or (tentative_theta_score == theta_scores[neighbor] and tentative_g_score < g_scores[neighbor]):

                g_scores[neighbor] = tentative_g_score
                theta_scores[neighbor] = tentative_theta_score
                previous_nodes[neighbor] = current_node

                # Compute f_score: g_score + heuristic, theta_score's heuristic is trivially assigned to zero
                f_score = tentative_g_score + heuristic(neighbor, end_node, coordinates_list)
                heapq.heappush(priority_queue, (tentative_theta_score, f_score, neighbor))

    # Backtrack to get the path from start_node to end_node
    path = []
    current = end_node
    while current is not None:
        path.append(current)
        current = previous_nodes[current]

    # Reverse path to get the correct order
    path = path[::-1]

    # Return an empty path if start_node and end_node are disconnected
    if path[0] != start_node:
        print("No feasible path exists!")
        return []

    return path, expanded_nodes


def hierarchical_planner(place_nodes, place_edges, room_adjacency_list, start_node_place, end_node_place, room_nodes_coordinates, n_classes, place_partition_info, root = None, Sem_Pred = 'MC', device = 'cpu', model = None):

    # Find parent nodes of the start and end places in the room layer.
    start_node_parent = place_nodes.parents[start_node_place]
    end_node_parent = place_nodes.parents[end_node_place]

    # Compute the high-level path in the rooms layer 
    room_path, expanded_rooms = class_ordered_a_star(room_adjacency_list, start_node_parent, end_node_parent, room_nodes_coordinates, n_classes, partition_info = place_partition_info, root = root, Sem_Pred = Sem_Pred, device = device, model = model)
    # print("High-level path (room layer):", room_path)

    # Compute the low-level path in the places layer 
    place_path, expanded_places = class_ordered_a_star(place_edges.adjacency_list, start_node_place, end_node_place, place_nodes.coordinates, n_classes, path_HL = room_path, nodes = place_nodes) 
    # print("Low-level path (place layer):", place_path)

    return room_path, place_path, expanded_rooms, expanded_places