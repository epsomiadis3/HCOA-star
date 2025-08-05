try:
    import spark_dsg as dsg
except ImportError:
    print("Spark_DSG not loaded")
import numpy as np
import pathlib
import pickle as pkl
from typing import List
from HCOAStar.dsg_processor.graph_utils import reassign_labels_by_geometry
from HCOAStar.dsg_processor.utils import EdgeInfo, NodeInfo, PartitionInfo

# Extract DSG Nodes
def extract_nodes(G: dsg.DynamicSceneGraph):
    print("Processing DSG (Rooms & 3D Places)...")

    # Initialization
    room_layer = G.get_layer(dsg.DsgLayers.ROOMS)
    room_index_map = [[], []] # [new indexing, DSG indexing]
    room_semantic_list, room_coordinates_list = [], []
    room_parent_list = []
    place_index_map = [[], []] # [new indexing, DSG indexing]
    place_semantic_list, place_coordinates_list = [], [] # Dimensions equal to the number of places
    place_parent_list = []
    place_partition = [] # list of subgraphs of the induced by the rooms
    building_index = 0
    room_index = 0
    place_index = 0

    place_layer = G.get_layer(dsg.DsgLayers.PLACES)

    if room_layer.num_nodes() <= 0:
        print("ERROR: No Rooms layers!")

    # Loop over rooms
    for node in room_layer.nodes:
        # Room Position
        node_pos = np.array(node.attributes.position)
        room_coordinates_list.append(node_pos)

        # Room Semantic Class
        room_semantic_list.append(node.attributes.semantic_label)

        # Indexing
        room_index_map[1].append(node.id.value)
        room_index_map[0].append(room_index)
        room_parent_list.append(building_index)

        subgraph = []
        child_nodes = node.children()

        # Loop over places
        for child_node in child_nodes:
            # Place Position
            node_pos = np.array(G.get_node(child_node).attributes.position)
            place_coordinates_list.append(node_pos)

            # Place Semantic Class
            place_label = G.get_node(child_node).attributes.semantic_label
            place_semantic_list.append(place_label)

            # Indexing
            place_index_map[1].append(child_node)
            place_index_map[0].append(place_index)
            place_parent_list.append(room_index)

            subgraph.append(place_index)
            place_index += 1

        place_partition.append(subgraph)
        room_index += 1

    place_nodes = NodeInfo(index_map=place_index_map, coordinates=place_coordinates_list, semantics=place_semantic_list, parents=place_parent_list)
    room_nodes = NodeInfo(index_map=room_index_map, coordinates=room_coordinates_list, semantics=room_semantic_list, parents=room_parent_list)

    print("   Done!")
    return place_nodes, room_nodes, place_partition


def find_semantic_classes_office(G, place_nodes, room_nodes, high_priority=2):
    # Initialize all nodes with high-priority class (here classes are in increasing order from least favorable to most)
    place_nodes.semantics = [high_priority] * len(place_nodes.semantics)
    room_nodes.semantics = [high_priority] * len(room_nodes.semantics)
    radius = 3
    com_label = 0

    # Assign label = 1 if in R(0)
    for k in range(len(place_nodes.coordinates)):
        if place_nodes.parents[k] == 0:
            place_nodes.semantics[k] = 1

    # Find computer positions
    object_layer = G.get_layer(dsg.DsgLayers.OBJECTS)
    comp_positions = [
        np.array(node.attributes.position)
        for node in object_layer.nodes
        if node.attributes.semantic_label == 10  # Label for computer
    ]

    # Reassign labels near computers
    for pos in comp_positions:
        place_nodes.semantics = reassign_labels_by_geometry(radius, pos, com_label, place_nodes.coordinates, place_nodes.semantics)

    return place_nodes.semantics, room_nodes.semantics


def find_semantic_classes_subway(G, place_nodes, room_nodes, high_priority = 1):
    # Initialize all nodes with high-priority class (here classes are in increasing order from least favorable to most)
    place_nodes.semantics = [high_priority] * len(place_nodes.semantics)
    room_nodes.semantics = [high_priority] * len(room_nodes.semantics)

    # Assign label = 0 if in R(5), assign label = 1 if in R(0)
    for k in range(len(place_nodes.coordinates)):
        if place_nodes.parents[k] == 5:
            place_nodes.semantics[k] = 0

        if place_nodes.parents[k] == 0:
            place_nodes.semantics[k] = 1


    return place_nodes.semantics, room_nodes.semantics


# Convert DSG index to new index
def get_node_index(node, place_index_map: List[List]):
    if isinstance(node, int):
        value = node
    else:
        value = node.id.value
    try:
        tmp = place_index_map[1].index(value)
        index = place_index_map[0][tmp]
        return index
    except:
        return None
    

def compute_adj_list(edges, edge_weights, edge_semantics, num_nodes):
    # Initialization
    adjacency_list = {i: [] for i in range(num_nodes)}

    # Create adjacency list
    for i, (u, v) in enumerate(edges):
        weight = edge_weights[i]
        semantic_class = edge_semantics[i]
        adjacency_list[u].append((v, weight, semantic_class))
        adjacency_list[v].append((u, weight, semantic_class))

    return adjacency_list


def extract_edges(G: dsg.DynamicSceneGraph, layer_index, nodes, node_partition, num_parents = 1):
    # Initialization
    node_index_map = nodes.index_map
    node_coordinates_list = nodes.coordinates
    node_semantics_list = nodes.semantics
    node_parent_list = nodes.parents
    num_nodes = len(node_coordinates_list)
    edge_list = []
    edge_weights = []
    edge_semantics = []
    edge_partition = [[] for _ in range(num_parents)]  # edges for each subgraph
    edge_weights_partition = [[] for _ in range(num_parents)]
    border_nodes = [0] * num_nodes  # List for features
    edges_DSG = G.get_layer(layer_index).edges

    for edge in edges_DSG:
        start_node, end_node = G.get_node(edge.source), G.get_node(edge.target)

        if start_node.id.value in node_index_map[1] and end_node.id.value in node_index_map[1]:
            start_index = get_node_index(start_node, node_index_map)
            end_index = get_node_index(end_node, node_index_map)

            # Compute the edge length (Euclidean distance)
            edge_len = np.linalg.norm(node_coordinates_list[start_index] - node_coordinates_list[end_index])

            # Determine edge semantic class
            edge_semantic_class_indiv = min(node_semantics_list[start_index], node_semantics_list[end_index])  # Use min for COA* comparison

            # Avoid duplicate edges and self-loops
            if ((start_index, end_index) not in edge_list and (end_index, start_index) not in edge_list and start_index != end_index):
                edge_list.append((start_index, end_index))
                edge_weights.append(edge_len)
                edge_semantics.append(edge_semantic_class_indiv)

                # Edge partitions
                parent_start = node_parent_list[start_index]
                parent_end = node_parent_list[end_index]

                if parent_start == parent_end:
                    # Edge numbering in each partition should start from 0
                    node_length = sum(len(sublist) for sublist in node_partition[:parent_start])
                    modified_edge = [(start_index - node_length, end_index - node_length)]
                    edge_partition[parent_start].append(modified_edge)
                    edge_weights_partition[parent_start].append(edge_len)
                else:
                    border_nodes[start_index] = 1
                    border_nodes[end_index] = 1

        else:
            print(f"Minor warning: Edge ({start_node.id.value}, {end_node.id.value}) not in initial layer No. {layer_index}, probably because one of the nodes did not have a parent")

    # Create adjacency list
    adjacency_list = compute_adj_list(edge_list, edge_weights, edge_semantics, num_nodes)

    return EdgeInfo(adjacency_list=adjacency_list, indexes=edge_list, weights=edge_weights, semantics=edge_semantics), edge_partition, edge_weights_partition, border_nodes


# Processing DSG: Edges
def generate_connectivity(G: dsg.DynamicSceneGraph, place_nodes, room_nodes, place_partition):
    print("Generating connectivity...")

    # Initialization
    interlayer_edge_list = []

    # ROOM EDGES
    room_edges, _, _, _ = extract_edges(G, dsg.DsgLayers.ROOMS, room_nodes, node_partition = [], num_parents = 1)

    # PLACES EDGES
    place_edges, place_edge_partition, place_edge_weights_partition, place_features = extract_edges(G, dsg.DsgLayers.PLACES, place_nodes, node_partition = place_partition, num_parents = len(room_nodes.coordinates))

    # INTERLAYER EDGES (for plots)
    for edge in G.interlayer_edges:
        source_layer = G.get_node(edge.source).layer
        target_layer = G.get_node(edge.target).layer
        if source_layer == 4 and target_layer == 3 : # ONLY ROOMS-PLACES
            start_node, end_node = G.get_node(edge.source), G.get_node(edge.target)
            if (start_node.id.value in room_nodes.index_map[1] and end_node.id.value in place_nodes.index_map[1]): # CHECK FOR REDUNCACES
                start_index, end_index = get_node_index(start_node, room_nodes.index_map), get_node_index(end_node, place_nodes.index_map)
                if (start_index, end_index) not in interlayer_edge_list and (end_index, start_index) not in interlayer_edge_list: # CHECK FOR DUPLICATES
                    interlayer_edge_list.append((start_index, end_index))
    
    interlayer_edge_list = list(interlayer_edge_list)
                
    print("   Done!")
    return place_edges, room_edges, interlayer_edge_list, place_edge_partition, place_edge_weights_partition, place_features


def add_connectivity(place_edges, room_edges, place_edge_partition, place_edge_weights_partition, border_nodes, place_nodes, place_partition, room_nodes, high_priority = 2):
    print("Making graph connected...")


    connection_pairs = [(2260, 2320), (2299, 2325), (2094, 2673), (2622, 181), (28, 147), (144, 267), (277, 2689), (215, 859), (281, 347)]

    for start_index, end_index in connection_pairs:
        edge_len = np.linalg.norm(place_nodes.coordinates[start_index] - place_nodes.coordinates[end_index])

        # Determine edge semantic class
        if isinstance(place_nodes.semantics[start_index], int) and isinstance(place_nodes.semantics[end_index], int):
            edge_semantic_class_indiv = min(place_nodes.semantics[start_index], place_nodes.semantics[end_index])  # Use min for COA* comparison
        else:
            edge_semantic_class_indiv = place_nodes.semantics[start_index]

        place_edges.indexes.append((start_index, end_index))
        place_edges.weights.append(edge_len)
        place_edges.semantics.append(edge_semantic_class_indiv)

        place_edges.adjacency_list[start_index].append((end_index, edge_len, edge_semantic_class_indiv))
        place_edges.adjacency_list[end_index].append((start_index, edge_len, edge_semantic_class_indiv))

        # Edge partitions
        parent_start = place_nodes.parents[start_index]
        parent_end = place_nodes.parents[end_index]

        if parent_start == parent_end:
            node_length = sum(len(sublist) for sublist in place_partition[:parent_start])
            modified_edge = [(start_index - node_length, end_index - node_length)]
            place_edge_partition[parent_start].append(modified_edge)
            place_edge_weights_partition[parent_start].append(edge_len)
        else:
            border_nodes[start_index] = 1
            border_nodes[end_index] = 1

            # Add inter-room edge if it does not already exist
            if (parent_start, parent_end) not in room_edges.indexes and (parent_end, parent_start) not in room_edges.indexes:
                room_edge_len = np.linalg.norm(room_nodes.coordinates[parent_start] - room_nodes.coordinates[parent_end])

                room_edges.indexes.append((parent_start, parent_end))
                room_edges.weights.append(room_edge_len)
                room_edges.semantics.append(high_priority)

                room_edges.adjacency_list[parent_start].append((parent_end, room_edge_len, high_priority))
                room_edges.adjacency_list[parent_end].append((parent_start, room_edge_len, high_priority))

    return place_edges, room_edges, place_edge_partition, place_edge_weights_partition, border_nodes
    


# Process Office Scene DSG
def convert_dsg_office(path_to_dsg, output_file="./data/graph_info.pkl"):
    # Load DSG
    path_to_dsg = pathlib.Path(path_to_dsg).expanduser().absolute()
    G = dsg.DynamicSceneGraph.load(str(path_to_dsg))
    print("DSG loaded")

    # Extract node info and induced subgraphs
    place_nodes, room_nodes, place_partition = extract_nodes(G = G)
    
    # Find node semantics
    place_nodes.semantics, room_nodes.semantics = find_semantic_classes_office(G, place_nodes, room_nodes, high_priority = 2)

    # Generate connectivity
    place_edges, room_edges, interlayer_edge_list, place_edge_partition, place_edge_weights_partition, place_features = generate_connectivity(G = G, place_nodes = place_nodes, room_nodes = room_nodes, place_partition = place_partition)
    
    place_features_partition = [[place_features[i] for i in sublist] for sublist in place_partition]
    place_semantics_partition = [[place_nodes.semantics[i] for i in sublist] for sublist in place_partition]

    place_partition_info = PartitionInfo(edge=place_edge_partition, edge_weights=place_edge_weights_partition, node_semantics=place_semantics_partition, node_features=place_features_partition)

    print("   Done!")
    print("Saving pkl file...")
    f = open(output_file, "wb")
    pkl.dump([place_nodes, place_edges, room_nodes, room_edges, interlayer_edge_list, place_partition_info], f)
    print("   Done!")



# Process Subway Scene DSG
def convert_dsg_subway(path_to_dsg, output_file="./data/graph_info.pkl"):
    # Load DSG
    path_to_dsg = pathlib.Path(path_to_dsg).expanduser().absolute()
    G = dsg.DynamicSceneGraph.load(str(path_to_dsg))
    print("DSG loaded")

    # Extract node info and induced subgraphs
    place_nodes, room_nodes, place_partition = extract_nodes(G = G)
    
    # Find node semantics
    place_nodes.semantics, room_nodes.semantics = find_semantic_classes_subway(G, place_nodes, room_nodes, high_priority = 2)

    # Generate connectivity
    place_edges, room_edges, interlayer_edge_list, place_edge_partition, place_edge_weights_partition, place_features = generate_connectivity(G = G, place_nodes = place_nodes, room_nodes = room_nodes, place_partition = place_partition)
    
    # Preprocessing Graph to make it connected
    place_edges, room_edges, place_edge_partition, place_edge_weights_partition, place_features = add_connectivity(place_edges, room_edges, place_edge_partition, place_edge_weights_partition, place_features, place_nodes, place_partition, room_nodes, high_priority = 2)

    place_features_partition = [[place_features[i] for i in sublist] for sublist in place_partition]
    place_semantics_partition = [[place_nodes.semantics[i] for i in sublist] for sublist in place_partition]

    place_partition_info = PartitionInfo(edge=place_edge_partition, edge_weights=place_edge_weights_partition, node_semantics=place_semantics_partition, node_features=place_features_partition)

    print("   Done!")
    print("Saving pkl file...")
    f = open(output_file, "wb")
    pkl.dump([place_nodes, place_edges, room_nodes, room_edges, interlayer_edge_list, place_partition_info], f)
    print("   Done!")

