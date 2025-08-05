import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_path(path, coordinates_list, ax):
    # Highlight the path
    x_prev, y_prev = None, None  # Initialize for intermediate nodes
    
    for idx, node in enumerate(path):
        x, y, z = coordinates_list[node]  # Unpack coordinates
        
        if idx == 0:  # Start node
            ax.scatter(x, y, color='black', s=140, label='Start', zorder=5)
            ax.text(x, y, 'S', color='white', ha='center', va='center', fontsize=12, weight='bold', zorder=6)
        elif idx == len(path) - 1:  # End node
            ax.scatter(x, y, color='black', s=140, label='Goal', zorder=5)
            ax.text(x, y, 'G', color='white', ha='center', va='center', fontsize=12, weight='bold', zorder=6)
            # Draw line to the previous node
            ax.plot([x_prev, x], [y_prev, y], 'k', linewidth = 3)    
        else:  # Intermediate nodes in the path
            ax.scatter(x, y, color='black', s=40, zorder=5)
            # Draw line to the previous node
            ax.plot([x_prev, x], [y_prev, y], 'k', linewidth = 3)    
        
        x_prev, y_prev = x, y


def plot_dsg_2D(coordinates_list, edge_list, label_list, label_colors, path=None, ax=None, x_lim=None, y_lim=None, margin=1, NodeSize=4, is_node_pruned=None, expanded_nodes = None):
    # Check if a standalone plot is required
    standalone = ax is None

    if standalone:
        figure = plt.figure(dpi=120)
        ax = figure.add_subplot()

    n_nodes = len(coordinates_list)
    if is_node_pruned == None:
        is_node_pruned = [False] * n_nodes

    # Plot edges
    for edge in edge_list:
        if is_node_pruned[edge[0]] == False and is_node_pruned[edge[1]] == False:
            pos1, pos2 = coordinates_list[edge[0]], coordinates_list[edge[1]]
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k-', linewidth=0.1, alpha=0.5)

    # Plot nodes
    index = 0
    for coordinate, label in zip(coordinates_list, label_list):
        if is_node_pruned[index] == False:
            color = label_colors[label]

            # Plot color for expanding nodes
            if expanded_nodes is not None:
                if expanded_nodes[index] == 1:
                    color = np.array([0.5, 0.5, 0.5])
                else:
                    color = np.array([0.9, 0.9, 0.9])

            ax.scatter(coordinate[0], coordinate[1], s=NodeSize, color=color)
        index += 1

    if path is not None:
        plot_path(path, coordinates_list, ax)

    # Set axis limits based on coordinates if not provided
    if x_lim is None:
        x_min, x_max = min(coord[0] for coord in coordinates_list), max(coord[0] for coord in coordinates_list)
        x_lim = [x_min, x_max]
    if y_lim is None:
        y_min, y_max = min(coord[1] for coord in coordinates_list), max(coord[1] for coord in coordinates_list)
        y_lim = [y_min, y_max]

    ax.set_xlim([x_lim[0] - margin, x_lim[1] + margin])
    ax.set_ylim([y_lim[0] - margin, y_lim[1] + margin])
    # ax.set_xlim([-20.100000381469727, 31.899999618530273])
    # ax.set_ylim([3.5, 48.29999923706055])
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()

    print(f"Figure dimensions: Width = {x_range[1] - x_range[0]}, Height = {y_range[1] - y_range[0]}")

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_aspect('equal')
    ax.set_frame_on(False)
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")

    if standalone:
        plt.savefig("scenario.png", bbox_inches='tight', pad_inches=0, dpi=300)
        plt.show()
    
    return x_lim, y_lim


def plot_dsg_2D_total(place_coordinates_list, place_edge_list, place_label_list, room_coordinates_list, room_edge_list, room_label_list, label_colors, is_place_node_pruned=None, place_path=None, room_path=None):
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=120)

    # Plot places on the left subplot
    x_lim, y_lim = plot_dsg_2D(place_coordinates_list, place_edge_list, place_label_list, label_colors, ax=axes[0], is_node_pruned = is_place_node_pruned, path = place_path)
    axes[0].set_title("Places")
    
    # Plot rooms on the right subplot, using the same limits
    plot_dsg_2D(room_coordinates_list, room_edge_list, room_label_list, label_colors, ax=axes[1], x_lim=x_lim, y_lim=y_lim, NodeSize = 30, path=room_path)
    axes[1].set_title("Rooms")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def plot_dsg_3D(place_coordinates_list, place_edge_list, room_coordinates_list, room_edge_list, interlayer_edge_list, place_color_list = None, room_color_list = None):
    figure = plt.figure(dpi=120)
    ax = figure.add_subplot(projection='3d')
    offset = 20
    if room_color_list is None:
        room_color_list = [np.array([0.0, 0.8, 0.0])] * len(room_coordinates_list)
    if place_color_list is None:
        place_color_list = [np.array([0.0, 0.8, 0.0])] * len(room_coordinates_list)

    # Plot place edges
    for edge in place_edge_list:
        pos1, pos2 = place_coordinates_list[edge[0]], place_coordinates_list[edge[1]]
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 'k-', linewidth=0.1, alpha=0.7)

    # Plot room edges
    for edge in room_edge_list:
        pos1, pos2 = room_coordinates_list[edge[0]], room_coordinates_list[edge[1]]
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2] + offset, pos2[2] + offset], 'k-', linewidth=1, alpha=0.7)

    # Plot interlayer edges
    for edge in interlayer_edge_list:
        pos1, pos2 = room_coordinates_list[edge[0]], place_coordinates_list[edge[1]]
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2] + offset, pos2[2]], 'k', linewidth=0.05, alpha=0.7, linestyle = (0, (1, 5)))

    # Plot place nodes
    for coordinate, color in zip(place_coordinates_list, place_color_list):
        ax.scatter(coordinate[0], coordinate[1], coordinate[2], s = 1, color = color)

    # Plot room nodes
    for coordinate, color in zip(room_coordinates_list, room_color_list):
        ax.scatter(coordinate[0], coordinate[1], coordinate[2] + offset, s = 10, color = color)

    # Set axis limits based on coordinates
    x_min, x_max = min(coord[0] for coord in place_coordinates_list), max(coord[0] for coord in place_coordinates_list)
    y_min, y_max = min(coord[1] for coord in place_coordinates_list), max(coord[1] for coord in place_coordinates_list)
    # z_min, z_max = min(coord[2] for coord in place_coordinates_list), max(coord[2] for coord in place_coordinates_list)

    ax.set_xlim([x_min - 5, x_max + 5])
    ax.set_ylim([y_min - 5, y_max + 5])
    # ax.set_zlim([z_min - 5, z_max + 5])

    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for 3D plot
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=50, azim=-70)  # elev=90 gives top-down view, azim=0 for "north" orientation

    plt.show()


def plot_dsg_2D_with_features(coordinates_list, edge_list, feature_list, label_list, label_colors, path=None, ax=None, x_lim=None, y_lim=None, margin=3, NodeSize=4):
    # Check if a standalone plot is required
    standalone = ax is None

    if standalone:
        figure = plt.figure(dpi=120)
        ax = figure.add_subplot()

    # Plot edges
    for edge in edge_list:
        pos1, pos2 = coordinates_list[edge[0]], coordinates_list[edge[1]]
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k-', linewidth=0.1, alpha=0.4)

    # Plot nodes
    for coordinate, label, feature in zip(coordinates_list, label_list, feature_list):
        # Set the color based on label
        color = label_colors[label]
        # color = '#006400' if feature == 1 else label_colors[label]  # Dark green if feature == 1
        # Choose a marker style
        marker = '*' if feature == 1 else 'o'  
        ax.scatter(coordinate[0], coordinate[1], s=NodeSize, color=color, marker=marker)

    if path is not None:
        plot_path(path, coordinates_list, ax)

    # Set axis limits based on coordinates if not provided
    if x_lim is None:
        x_min, x_max = min(coord[0] for coord in coordinates_list), max(coord[0] for coord in coordinates_list)
        x_lim = [x_min, x_max]
    if y_lim is None:
        y_min, y_max = min(coord[1] for coord in coordinates_list), max(coord[1] for coord in coordinates_list)
        y_lim = [y_min, y_max]

    ax.set_xlim([x_lim[0] - margin, x_lim[1] + margin])
    ax.set_ylim([y_lim[0] - margin, y_lim[1] + margin])

    ax.set_aspect('equal')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    if standalone:
        plt.show()
    
    return x_lim, y_lim


def plot_dsg_2D_with_features_subgraphs(place_coordinates_partition, place_partition, edge_partition, place_features_partition, place_label_partition, label_colors):
    num_partitions = len(place_partition)
    
    # Create a figure with 2 rows and 4 columns (4 on top, 3 on bottom)
    fig, axes = plt.subplots(2, 4, figsize=(8, 4), dpi=120)
    
    # Flatten axes to easily iterate through them (since we have a 2D array of axes)
    axes = axes.flatten()

    # Plot places on each subplot
    for index in range(num_partitions):  # Corrected loop range
        x_lim, y_lim = plot_dsg_2D_with_features(place_coordinates_partition[index], edge_partition[index], place_features_partition[index], place_label_partition[index], label_colors, ax=axes[index])
        axes[index].set_title(f"Room {index}")  # Corrected title formatting

    # Turn off axes for any unused subplots (if num_partitions < 8)
    for index in range(num_partitions, len(axes)):
        axes[index].axis('off')
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


# Visualize connected components
def plot_dsg_components(coordinates_list, edge_list):
    fig, ax = plt.subplots(figsize=(8, 8))
    # Create a graph from the edge list
    G = nx.Graph()
    G.add_edges_from(edge_list)
    
    # Find connected components of the graph
    connected_components = list(nx.connected_components(G))
    
    # Create a list of colors, each component will have its own color
    component_colors = plt.cm.get_cmap('tab10', len(connected_components))

    # Assign a color to each node based on its component
    node_color_map = {}
    for idx, component in enumerate(connected_components):
        color = component_colors(idx)  # Get a unique color from the colormap
        for node in component:
            node_color_map[node] = color

    # Initialize node list for plotting
    node_list = []

    # Plot edges
    for edge in edge_list:
        pos1, pos2 = coordinates_list[edge[0]], coordinates_list[edge[1]]
        # if pos1[2] > -7.5: # different floor for subway scene
        #     continue
    
        plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k-', linewidth=0.2, alpha=0.7)

    # Plot nodes with colors based on their connected component
    for i, coordinate in enumerate(coordinates_list):
        # if coordinate[2] > -7.5: # different floor for subway scene
        #     continue

        node_color = node_color_map.get(i, (0.5, 0.5, 0.5))  # Default to gray if no color assigned
        node = plt.Circle(coordinate, 0.2, color=node_color)
        node_list.append(node)
        ax.add_artist(node)

    # Set plot limits based on node coordinates
    x_min, x_max = min([coordinate[0] for coordinate in coordinates_list]), max([coordinate[0] for coordinate in coordinates_list])
    y_min, y_max = min([coordinate[1] for coordinate in coordinates_list]), max([coordinate[1] for coordinate in coordinates_list])

    # ax.set_xlim([x_min - 20, x_max + 20])
    # ax.set_ylim([y_min - 20, y_max + 20])
    ax.axis('equal')

    plt.show()