import numpy as np
from matplotlib import pyplot, cm
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from scipy.cluster import hierarchy as shc
from scipy.spatial.distance import pdist, squareform
from shapely.geometry import Polygon, mapping
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

def plot_multiple_3d_points(x, y, z, x_, y_, z_):
    # Convert inputs to numpy arrays for easier manipulation
    x, y, z = np.array(x), np.array(y), np.array(z)
    x_, y_, z_ = np.array(x_), np.array(y_), np.array(z_)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(x, y, z, color='red', s=5)
    ax.scatter(x_, y_, z_, color='blue', s=5)

    # Calculate the overall range to use for the axis limits
    all_x = np.concatenate([x, x_])
    all_y = np.concatenate([y, y_])
    all_z = np.concatenate([z, z_])

    max_range = np.array([all_x.max() - all_x.min(), all_y.max() - all_y.min(), all_z.max() - all_z.min()]).max() / 2.0

    mid_x = (all_x.max() + all_x.min()) * 0.5
    mid_y = (all_y.max() + all_y.min()) * 0.5
    mid_z = (all_z.max() + all_z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Set labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Plot of Multiple 3D Points')

    plt.show()

def quaternion_to_direction_vector(qx, qy, qz, qw):
    """
    Convert a quaternion to a direction vector, assuming the forward direction is along the x-axis.
    """
    # Rotating the standard x-axis unit vector by the quaternion
    # Quaternion vector multiplication (q * v * q^-1)
    # For the x-axis unit vector, this simplifies since the vector part is (1, 0, 0)
    v = np.array([1, 0, 0])
    q = np.array([qw, qx, qy, qz])
    q_conj = np.array([qw, -qx, -qy, -qz])
    v = quaternion_multiply(quaternion_multiply(q, np.concatenate(([0], v))), q_conj)
    return v[1:]  # Return the vector part

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return np.array([w, x, y, z])

# def flip_rot_matrix(rot_matrix):
#     flip_matrix = np.array([
#         [-1,0,0],
#         [0,1,0],
#         [0,0,-1]
#     ])
#     return np.dot(rot_matrix, flip_matrix)

# def quaternion_from_matrix(matrix):
#     m00,m01,m02,m10,m11,m12,m20,m21,m22 = matrix.flatten()
#     q0 = np.sqrt(max(0, 1 + m00 + m11 + m22)) /2
#     q1 = np.sqrt(max(0, 1 + m00 - m11 - m22)) /2
#     q2 = np.sqrt(max(0, 1 - m00 + m11 - m22)) /2
#     q3 = np.sqrt(max(0, 1 - m00 - m11 + m22)) /2

#     q1 = q1 * np.sign(m21 - m12)
#     q2 = q2 * np.sign(m02 - m20)
#     q3 = q3 * np.sign(m10 - m01)
#     return np.array([q1, q2, q3, q0])

def plot_poses(pose_array, length = 0.1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract positions and quaternions from the pose array
    positions = pose_array[:, :3]
    quaternions = pose_array[:, 3:]
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    # Plot the positions as red points
    ax.scatter(x,y,z, color='red', s=10)

    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Plot the orientation for each pose
    for pos, quat in zip(positions, quaternions):
        direction = quaternion_to_direction_vector(quat[0], quat[1], quat[2], quat[3])
        ax.quiver(pos[0], pos[1], pos[2], direction[0], direction[1], direction[2], length=length, normalize=True)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Pose Orientation')

    # Set aspect of the plot to equal
    ax.set_box_aspect([1,1,1])  # Aspect ratio is 1:1:1

    # Show the plot
    plt.show()

# used only for show purpose
def calculate_viewpoints_raw(pose_array, distance):
    # Extract positions and quaternions from the pose array
    positions = pose_array[:, :3]
    quaternions = pose_array[:, 3:]

    # Prepare an array to hold the new points
    new_points = np.zeros_like(positions)

    # Calculate new point for each pose
    for i, (pos, quat) in enumerate(zip(positions, quaternions)):
        direction = quaternion_to_direction_vector(quat[0], quat[1], quat[2], quat[3])
        new_points[i] = pos + direction * distance  # Move along the direction by the specified distance

    return new_points

def calculate_viewpoints(pose_array, distance, angle_tolerance=45, z_min=0.1):
    # Calculate the cosine of the angle tolerance for comparison
    cos_tolerance = np.cos(np.radians(angle_tolerance))
    
    # The negative z-axis vector
    down_vector = np.array([0, 0, -1])

    new_points = []
    for pose in pose_array:
        position = pose[:3]
        quaternion = pose[3:]
        
        direction = quaternion_to_direction_vector(quaternion[0], quaternion[1], quaternion[2], quaternion[3])

        # Check if the direction vector points too much downward
        cos_angle = np.dot(direction, down_vector) / np.linalg.norm(direction)
        if cos_angle < cos_tolerance:  # If the direction is within the allowed tolerance
            new_point = position + direction * distance
            if new_point[2] >= z_min:  # Check if the z-value of the new point is above the minimum
                # q = quaternion
                # rot_matrix = np.array([
                #     [1 - 2*q[1]**2 - 2*q[2]**2, 2*q[0]*q[1] - 2*q[2]*q[3], 2*q[0]*q[2] + 2*q[1]*q[3]],
                #     [2*q[0]*q[1] + 2*q[2]*q[3], 1 - 2*q[0]**2 - 2*q[2]**2, 2*q[1]*q[2] - 2*q[0]*q[3]],
                #     [2*q[0]*q[2] - 2*q[1]*q[3], 2*q[1]*q[2] + 2*q[0]*q[3], 1 - 2*q[0]**2 - 2*q[1]**2]
                # ])
                # flip_matrix = flip_rot_matrix(rot_matrix)
                # flip_quat = quaternion_from_matrix(rot_matrix)
                # flip_quat /= np.linalg.norm(flip_quat)
                
                combined_position_orientation = np.concatenate((new_point, quaternion))
                new_points.append(combined_position_orientation)

    return np.array(new_points)

def viewpoint_clusters(viewpoints, d_cluster=1.7):
    """
    Takes points and uses complete hierarchial clustering to return cluster centers in 2D
    \nReturns:
    cluster_groups: a list of group numbers for each viewpoint
    cluster_centers: a 2D XY array of cluster centers
    """
    distMatrix = pdist(viewpoints)
    Z = shc.complete(distMatrix)
    cluster_groups = shc.fcluster(Z, d_cluster, criterion='distance')

    # n_clusters = max(cluster_groups)
    # cluster_centers = np.zeros((n_clusters, 2))
    # for c in range(n_clusters):
    #     group = cluster_groups == c+1
    #     view_group = viewpoints[group, 0:2]
    #     cluster_centers[c] = np.median(view_group, axis=0)

    return cluster_groups #, cluster_centers

""" plot model and all clusters viewpoints   """
def plot_viewpoints_cluster(viewpoints, cluster_groups):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # Use the tab10 colormap for up to 10 distinct groups
    colormap = plt.get_cmap('tab10')
    # Determine the number of unique groups and fetch distinct colors
    num_groups = len(np.unique(cluster_groups))
    colors = [colormap(i) for i in range(num_groups)]

    # Generate a color map or define your own colors
    colors = plt.cm.jet(np.linspace(0, 1, len(np.unique(cluster_groups))))
    # print(colors)

    for group, color in zip(np.unique(cluster_groups), colors):
        # Select points belonging to the current group
        indices = cluster_groups == group
        group_points = viewpoints[indices]
        
        # Plot the points with the chosen color
        ax.scatter(group_points[:, 0], group_points[:, 1], group_points[:, 2], color=color, label=group, s=5)
    # ax.scatter(model_points[:,0], model_points[:,1], model_points[:,2], color='grey', s=5)
    # ax.scatter(cluster_centers[:,0], cluster_centers[:,1], np.zeros(cluster_centers.shape[0]), color='grey', s=5)

    x = viewpoints[:,0]
    y = viewpoints[:,1]
    z = viewpoints[:,2]
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Viewpoints by Group')

    # Legend
    ax.legend(title="Groups")

    # Show plot
    plt.show()

# def create_boundary(pose_array, offset_distance = 0.3, plot=True, robot_z = 0.5):
#     points_2d = pose_array[:, :2]
#     # Create a Polygon from 2D points
#     polygon = Polygon(points_2d)

#     # Generate the offset polygon (buffered polygon)
#       # Distance by which the polygon should be expanded
#     offset_polygon = polygon.buffer(offset_distance)
    
#     # x, y = offset_polygon.exterior.xy

#     # Simplify the offset polygon
#     simplification_tolerance = 0.005  # Change this value to see different levels of simplification
#     simplified_polygon = offset_polygon.simplify(simplification_tolerance, preserve_topology=True)

#     x, y = simplified_polygon.exterior.xy
#     z = np.full((len(x), 1), robot_z).squeeze()
#     print(z.shape, np.zeros(len(x)).shape)
#     if plot:
#         # Plotting the original and offset polygons
#         fig, ax = plt.subplots()

        
#         ax.scatter(points_2d[:,0], points_2d[:,1], alpha=0.5, fc='red', label='Original Polygon',s=5)    
#         # ax.scatter(x, y, alpha=0.5, fc='blue', label='Offset Polygon', s=5)    
#         ax.scatter(x, y, alpha=0.8, fc='blue', label='Simplified Polygon', s=5)

#         ax.set_aspect('equal', 'box')
#         ax.set_title('Polygon and Its Offset')
#         ax.legend()
#         plt.show()
#     return np.array([x,y,z]).T

def check_bound_clusters_optimized(viewpoints, cluster_group, bound, reach_max, reach_min):
    num_clusters = np.unique(cluster_group).size
    results = []
    best_bound_points = []  # To store the best bound point for each cluster

    for i in range(1,num_clusters):
        cluster_points = viewpoints[cluster_group == i]

        # Calculate distances between each bound point and each cluster point
        # Result shape: (num_bound_points, num_cluster_points)
        dist_matrix = np.sqrt(((bound[:, np.newaxis, :] - cluster_points) ** 2).sum(axis=2))

        # Check distances within specified min and max range
        within_range = (dist_matrix < reach_max) & (dist_matrix > reach_min)

        # Count how many points each bound point can reach within the range
        reachability_count = within_range.sum(axis=1)

        # Find the bound point that reaches the most cluster points
        max_reachable_idx = np.argmax(reachability_count)
        max_reachable = reachability_count[max_reachable_idx]
        unreachable_count = cluster_points.shape[0] - max_reachable

        # Store the best bound point for this cluster
        best_bound_point = bound[max_reachable_idx]
        best_bound_points.append(best_bound_point)

        # Gather data for reachability
        reachable_indices = np.where(within_range[max_reachable_idx])[0]
        reached_points = cluster_points[reachable_indices]
        unreachable_points = cluster_points[~within_range[max_reachable_idx]]

        results.append({
            'cluster': i,
            'best_bound_point': bound[max_reachable_idx],
            'reachable_count': max_reachable,
            'unreachable_count': unreachable_count,
            'reachable_points': reached_points,
            'unreachable_points': unreachable_points
        })

    return results, np.array(best_bound_points)

def plot_results(results, positions):
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    # Plotting
    for result in results:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x,y,z, color='grey', label='model')

        # Plot reachable points
        if result['reachable_points'].size > 0:
            ax.scatter(*result['reachable_points'].T, color='green', label='Reachable Points')
        
        # Plot unreachable points
        if result['unreachable_points'].size > 0:
            ax.scatter(*result['unreachable_points'].T, color='red', label='Unreachable Points')
        
        # Plot best bound point
        ax.scatter(*result['best_bound_point'], color='blue', marker='^', s=100, label='Best Bound Point')
        ax.set_aspect('equal', 'box')
        ax.set_title(result['cluster'])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

    plt.show()

def solve_tsp_2d(points, start_point, plot=True):
    # Calculate the distance matrix including the start point
    all_points = np.vstack([start_point, points])  # Stack the start point with other points
    dist_matrix = cdist(all_points, all_points)  # Compute the distance matrix

    num_points = all_points.shape[0]
    visited = np.zeros(num_points, dtype=bool)
    tour = [0]  # Start the tour at the first point (our start point)
    visited[0] = True

    current_point = 0

    # Implementing nearest-neighbor heuristic
    while np.sum(visited) < num_points:
        # Find the nearest unvisited point
        distances_to_current = dist_matrix[current_point]
        distances_to_current[visited] = np.inf  # Set visited points' distances to infinity
        next_point = np.argmin(distances_to_current)
        visited[next_point] = True
        tour.append(next_point)
        current_point = next_point

    # Return to the starting point
    tour.append(0)

    # Plotting if requested
    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(all_points[:, 0], all_points[:, 1], color='blue')  # plot points
        plt.plot(all_points[tour, 0], all_points[tour, 1], 'r-')  # plot tour
        plt.scatter(start_point[0], start_point[1], color='red', marker='s')  # mark the starting point
        plt.title('2D TSP Solution from Custom Start Point')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.show()

    return tour, all_points

def create_boundary(pose_array, offset_distance=0.3, plot=True, robot_z=0.3):
    points_2d = pose_array[:, :2]
    polygon = Polygon(points_2d)
    offset_polygon = polygon.buffer(offset_distance)
    simplification_tolerance = 0.005
    simplified_polygon = offset_polygon.simplify(simplification_tolerance, preserve_topology=True)
    
    x, y = simplified_polygon.exterior.xy
    z = np.full((len(x),), robot_z)
    
    if plot:
        fig, ax = plt.subplots()
        ax.scatter(points_2d[:,0], points_2d[:,1], alpha=0.5, color='red', label='Original Polygon')
        ax.scatter(x, y, alpha=0.8, color='blue', label='Simplified Polygon')
        ax.set_aspect('equal', 'box')
        ax.set_title('Polygon and Its Offset')
        ax.legend()
        plt.show()
        
    return np.column_stack((x, y, z))

# def slice_3d_viewpoints(viewpoints, pose_array, offset_distances, reach_min, reach_max, plot=False, sub_clusters=1):
#     data_for_pca = viewpoints[:, :3]
#     pca = PCA(n_components=2)
#     viewpoints_pca = pca.fit_transform(data_for_pca)
#     median_value = np.median(viewpoints_pca[:, 1])
#     indices = [np.where(viewpoints_pca[:, 1] <= median_value)[0], np.where(viewpoints_pca[:, 1] > median_value)[0]]

#     cluster_data = []
#     for original_indices in indices:
#         if sub_clusters > 1:
#             cluster = viewpoints[original_indices, :3]
#             kmeans = KMeans(n_clusters=sub_clusters, random_state=0).fit(cluster)
#             sub_labels = kmeans.labels_
#             for i in range(sub_clusters):
#                 sub_cluster_indices = original_indices[sub_labels == i]
#                 sub_cluster_data = viewpoints[sub_cluster_indices, :3]
#                 cluster_data.append(sub_cluster_data)
#         else:
#             sub_cluster_data = viewpoints[original_indices, :3]
#             cluster_data.append(sub_cluster_data)

#     for sub_cluster in cluster_data:
#         best_reach_count = 0
#         best_reach_point = None
#         for offset_distance in offset_distances:
#             bound = create_boundary(pose_array, offset_distance, plot=False)
#             distances = np.linalg.norm(sub_cluster[:, np.newaxis, :] - bound[np.newaxis, :, :], axis=2)
#             reachable = (distances <= reach_max) & (distances >= reach_min)
#             reach_count = np.sum(reachable, axis=0)
#             if np.max(reach_count) > best_reach_count:
#                 best_reach_count = np.max(reach_count)
#                 best_reach_point = bound[np.argmax(reach_count)]
#         if plot:
#                 fig = plt.figure()
#                 ax = fig.add_subplot(111, projection='3d')
#                 reachable_mask = np.any(reachable, axis=1)
#                 ax.scatter(*sub_cluster[reachable_mask].T, color='green', marker='o', label='Reachable Points')
#                 ax.scatter(*sub_cluster[~reachable_mask].T, color='red', marker='x', label='Non-Reachable Points')
#                 ax.scatter(*best_reach_point, color='gold', marker='^', s=100, label='Best Reach Point')
#                 ax.set_xlabel('X Axis')
#                 ax.set_ylabel('Y Axis')
#                 ax.set_zlabel('Z Axis')
#                 ax.set_title(f'Cluster Analysis - Best Offset {offset_distance} with Reach Count {best_reach_count}')
#                 ax.legend()
#                 plt.show()

#         print(f"Cluster with the best reach point at {best_reach_point} reaching {best_reach_count} points.")

def calculate_distance_matrix(points):
    """Calculate the Euclidean distance matrix for a set of points."""
    return np.linalg.norm(points[:, None] - points[None, :], axis=-1)

def nearest_neighbor_tsp(points, start_index=0, plot=False, model=None):
    """Solves the TSP using the nearest neighbor heuristic."""
    n_points = len(points)
    unvisited = set(range(n_points))
    unvisited.remove(start_index)
    tour = [start_index]
    current_index = start_index

    while unvisited:
        next_index = min(unvisited, key=lambda index: calculate_distance_matrix(points)[current_index, index])
        unvisited.remove(next_index)
        tour.append(next_index)
        current_index = next_index

    # Optionally close the tour
    tour.append(start_index)  # Return to the starting point

    if plot:
        # Plot the tour
        plt.figure()
        plt.plot(points[:, 0], points[:, 1], 'o', markerfacecolor='blue', markersize=10, label='Centroids')
        # print(points)
        # Highlight the origin
        plt.plot(points[start_index, 0], points[start_index, 1], 'o', color='red', markersize=12, label='Origin')
        for i in range(len(tour) - 1):
            start_point = points[tour[i]]
            end_point = points[tour[i + 1]]
            plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k-')
        if model is not None:
            plt.scatter(model[:, 0], model[:, 1])
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title('TSP Nearest Neighbor Tour')
        plt.legend()
        plt.grid(True)
        plt.show()

    return tour

def nearest_neighbor_tsp_3d(start_point, points):
    """Compute the TSP path using the nearest neighbor heuristic and return both forward and backward paths."""
    path = [start_point]
    visited = np.zeros(len(points), dtype=bool)
    current_point = start_point

    while not np.all(visited):
        distances = np.linalg.norm(points[:, :3] - current_point[:3], axis=1)
        distances[visited] = np.inf  # Ignore already visited points
        next_point_index = np.argmin(distances)
        visited[next_point_index] = True
        current_point = points[next_point_index, :]
        path.append(current_point)

    forward_path = np.array(path)
    backward_path = forward_path[::-1]  # Reverse the path to create the backward path

    return forward_path, backward_path

def plot_tsp_paths(l1, l2, plot=False):
    path = []  # List to store backward paths without the last row

    for idx, (points, start_point) in enumerate(zip(l1, l2)):
        forward_path, backward_path = nearest_neighbor_tsp_3d(start_point, points)
        # Append backward path without the last row
        path.append(backward_path[:-1])

        if plot:
            if idx == 0:  # Initialize plotting only if it's the first loop iteration and plotting is needed
                fig = plt.figure(figsize=(12, 16))

            # Plot forward path
            ax1 = fig.add_subplot(len(l1), 2, 2 * idx + 1, projection='3d')
            ax1.plot(forward_path[:, 0], forward_path[:, 1], forward_path[:, 2], 'g-o', label='Forward Path')
            ax1.scatter([start_point[0]], [start_point[1]], [start_point[2]], color='red', s=100, label='Start')
            ax1.set_title('Forward TSP Path' + str(idx + 1))
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.legend()

            # Plot backward path
            ax2 = fig.add_subplot(len(l1), 2, 2 * idx + 2, projection='3d')
            ax2.plot(backward_path[:, 0], backward_path[:, 1], backward_path[:, 2], 'b-o', label='Backward Path')
            ax2.scatter([start_point[0]], [start_point[1]], [start_point[2]], color='red', s=100, label='Start')
            ax2.set_title('Backward TSP Path'+ str(idx + 1))
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.legend()

    if plot:
        plt.show()
        
    return path

def slice_3d_viewpoints(viewpoints, bound, reach_min, reach_max, model=None, plot=False, sub_clusters=1):
    """
    Processes the viewpoints data to find clusters, determine reachable and non-reachable points
    from the bound set, and optionally plots detailed views with model data.
    Returns clusters with only reachable points and their corresponding max reach points.
    """
    if not isinstance(viewpoints, np.ndarray) or viewpoints.shape[1] < 7:
        raise ValueError("Viewpoints must be a numpy array with at least 7 columns")
    if not isinstance(bound, np.ndarray) or bound.shape[1] != 3:
        raise ValueError("Bound must be a numpy array with 3 columns")

    # PCA on the first three columns
    data_for_pca = viewpoints[:, :3]
    pca = PCA(n_components=2)
    viewpoints_pca = pca.fit_transform(data_for_pca)
    median_value = np.median(viewpoints_pca[:, 1])
    indices_1 = np.where(viewpoints_pca[:, 1] <= median_value)[0]
    indices_2 = np.where(viewpoints_pca[:, 1] > median_value)[0]

    cluster_data = []
    max_reach_points = []
    for original_indices in [indices_1, indices_2]:
        if sub_clusters > 1:
            cluster = viewpoints[original_indices, :3]
            kmeans = KMeans(n_clusters=sub_clusters, random_state=0).fit(cluster)
            sub_labels = kmeans.labels_
            for i in range(sub_clusters):
                sub_cluster_indices = original_indices[sub_labels == i]
                sub_cluster_data = viewpoints[sub_cluster_indices, :]
                cluster_data.append(sub_cluster_data)
        else:
            sub_cluster_data = viewpoints[original_indices, :]
            cluster_data.append(sub_cluster_data)

    reachable_cluster_data = []
    for i, sub_cluster in enumerate(cluster_data):
        distances = np.linalg.norm(sub_cluster[:, :3][:, np.newaxis, :] - bound[np.newaxis, :, :], axis=2)
        reachable = (distances <= reach_max) & (distances >= reach_min)
        max_reach_index = np.argmax(np.sum(reachable, axis=0))
        max_reach_point = bound[max_reach_index]
        max_reach_points.append(max_reach_point)

        # Filter to keep only reachable points
        reachable_sub_cluster = sub_cluster[np.any(reachable, axis=1), :7]
        reachable_cluster_data.append(reachable_sub_cluster)

        # print(f"Cluster {i+1}: Best bound point {max_reach_point} reaches {np.sum(reachable[:, max_reach_index])} out of {len(sub_cluster)} points.")

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(*reachable_sub_cluster[:,:3].T, color='green', marker='o', label='Reachable Points')
            ax.scatter(*max_reach_point, color='gold', marker='^', s=100, label='Max Reach Point')
            if model is not None:
                ax.scatter(*model.T, color='grey', marker='p', label='Model Data')

            ax.set_title('Cluster '+(i+1) + 'Analysis - Reachable Points')
            ax.legend()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_aspect('equal', 'box')
            plt.show()

    return reachable_cluster_data, max_reach_points
""" the one"""
# def slice_3d_viewpoints(viewpoints, bound, reach_min, reach_max, model=None, plot=False, sub_clusters=1):
#     """
#     Processes the viewpoints data to find clusters, determine reachable and non-reachable points
#     from the bound set, and optionally plots detailed views with model data.
#     """
#     if not isinstance(viewpoints, np.ndarray) or viewpoints.shape[1] < 7:
#         raise ValueError("Viewpoints must be a numpy array with at least 7 columns")
#     if not isinstance(bound, np.ndarray) or bound.shape[1] != 3:
#         raise ValueError("Bound must be a numpy array with 3 columns")

#     # PCA on the first three columns
#     data_for_pca = viewpoints[:, :3]
#     pca = PCA(n_components=2)
#     viewpoints_pca = pca.fit_transform(data_for_pca)
#     median_value = np.median(viewpoints_pca[:, 1])
#     indices_1 = np.where(viewpoints_pca[:, 1] <= median_value)[0]
#     indices_2 = np.where(viewpoints_pca[:, 1] > median_value)[0]

#     cluster_data = []
#     for original_indices in [indices_1, indices_2]:
#         if sub_clusters > 1:
#             cluster = viewpoints[original_indices, :3]
#             kmeans = KMeans(n_clusters=sub_clusters, random_state=0).fit(cluster)
#             sub_labels = kmeans.labels_
#             for i in range(sub_clusters):
#                 sub_cluster_indices = original_indices[sub_labels == i]
#                 sub_cluster_data = viewpoints[sub_cluster_indices, :]
#                 cluster_data.append(sub_cluster_data)
#         else:
#             sub_cluster_data = viewpoints[original_indices, :]
#             cluster_data.append(sub_cluster_data)

#     max_reach_points = []
#     # Process each sub-cluster
#     for i, sub_cluster in enumerate(cluster_data):
#         distances = np.linalg.norm(sub_cluster[:, :3][:, np.newaxis, :] - bound[np.newaxis, :, :], axis=2)
#         reachable = (distances <= reach_max) & (distances >= reach_min)
#         max_reach_index = np.argmax(np.sum(reachable, axis=0))
#         max_reach_point = bound[max_reach_index]
#         max_reach_points.append(max_reach_point)
#         # Print details for each cluster
#         print(f"Cluster {i+1}: Best bound point {max_reach_point} reaches {np.sum(reachable[:, max_reach_index])} out of {len(sub_cluster)} points.")
#         # Detailed plot for each cluster
#         if plot:
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d')
#             reachable_mask = np.any(reachable, axis=1)
#             print(reachable)
#             # print(sub_cluster[reachable_mask, :3].T.shape)

#             ax.scatter(*sub_cluster[reachable_mask, :3].T, color='green', marker='o', label='Reachable Points')
#             ax.scatter(*sub_cluster[~reachable_mask, :3].T, color='red', marker='x', label='Non-Reachable Points')
#             ax.scatter(*max_reach_point, color='gold', marker='^', s=100, label='Max Reach Point')
#             if model is not None:
#                 ax.scatter(*model.T, color='grey', marker='p', label='Model Data')

#             ax.set_title(f'Cluster {i+1} Analysis - Reachable vs Non-Reachable')
#             ax.legend()
#             ax.set_xlabel('X')
#             ax.set_ylabel('Y')
#             ax.set_zlabel('Z')
#             ax.set_aspect('equal', 'box')
#             plt.show()

#     return cluster_data,max_reach_points

# def slice_3d_viewpoints(viewpoints, bound, reach_min, reach_max, model=None, plot=False, sub_clusters=1):
#     """
#     Slices a set of 3D points into clusters, finds the point in 'bound' for each cluster with the most points reached
#     within specified reach distances, prints details, optionally plots the results with model data, and plots the best reach points.
    
#     Parameters:
#     - viewpoints (np.array): A numpy array of shape (n_samples, 7) where only the first three columns are used for clustering.
#     - bound (np.array): A numpy array of shape (m_samples, 3) containing bound points.
#     - reach_min (float): Minimum distance for a point to be considered reachable.
#     - reach_max (float): Maximum distance for a point to be considered reachable.
#     - model (np.array): Optional. A numpy array of shape (m_samples, 3) to plot in grey.
#     - plot (bool): If True, plots the clusters with equal axis ratios.
#     - sub_clusters (int): Number of sub-clusters to divide each of the two main clusters into.

#     Returns:
#     - list: A list containing tuples with each tuple containing a sub-cluster data, best bound point, and reach count.
#     """
#     if not isinstance(viewpoints, np.ndarray) or viewpoints.shape[1] < 7:
#         raise ValueError("Viewpoints must be a numpy array with at least 7 columns")

#     if not isinstance(bound, np.ndarray) or bound.shape[1] != 3:
#         raise ValueError("Bound must be a numpy array with 3 columns")

#     # Use only the first three columns for PCA
#     data_for_pca = viewpoints[:, :3]

#     # PCA to find the principal components
#     pca = PCA(n_components=2)
#     viewpoints_pca = pca.fit_transform(data_for_pca)
    
#     # Find median along the second principal component to split the data
#     median_value = np.median(viewpoints_pca[:, 1])
    
#     # Split the data based on the median of the second component
#     indices_1 = np.where(viewpoints_pca[:, 1] <= median_value)[0]
#     indices_2 = np.where(viewpoints_pca[:, 1] > median_value)[0]

#     # Further cluster each main cluster into sub-clusters if requested
#     cluster_data = []
#     for original_indices in [indices_1, indices_2]:
#         if sub_clusters > 1:
#             cluster = viewpoints[original_indices, :3]
#             kmeans = KMeans(n_clusters=sub_clusters, random_state=0).fit(cluster)
#             sub_labels = kmeans.labels_
#             for i in range(sub_clusters):
#                 sub_cluster_indices = original_indices[sub_labels == i]
#                 sub_cluster_data = viewpoints[sub_cluster_indices, :]
#                 cluster_data.append(sub_cluster_data)
#         else:
#             sub_cluster_data = viewpoints[original_indices, :]
#             cluster_data.append(sub_cluster_data)

#     # Calculate most reached bound point for each sub-cluster
#     cluster_results = []
#     for sub_cluster in cluster_data:
#         reach_counts = []
#         for bound_point in bound:
#             distances = np.linalg.norm(sub_cluster[:, :3] - bound_point, axis=1)
#             reach_count = np.sum((distances >= reach_min) & (distances <= reach_max))
#             reach_counts.append(reach_count)
#         most_reached_index = np.argmax(reach_counts)
#         most_reached_point = bound[most_reached_index]
#         cluster_results.append((sub_cluster, most_reached_point, reach_counts[most_reached_index]))

#         # Print details about the best point for each cluster
#         print(f"Best point for cluster with {len(sub_cluster)} points: {most_reached_point}")
#         print(f"Can reach {reach_counts[most_reached_index]} out of {len(sub_cluster)} points")

#     # Optional plotting
#     if plot:
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         colors = plt.cm.get_cmap('tab20', 20).colors
#         for i, (sub_cluster, best_point, _) in enumerate(cluster_results):
#             ax.scatter(sub_cluster[:, 0], sub_cluster[:, 1], sub_cluster[:, 2], color=colors[i % 20], label=f'Sub-cluster {i+1}')
#             ax.scatter(*best_point, color='black', marker='x', s=100, label=f'Best Point Cluster {i+1}')  # Mark the best point
        
#         if model is not None:
#             if not isinstance(model, np.ndarray) or model.shape[1] != 3:
#                 raise ValueError("Model data must be a numpy array with shape (m, 3)")
#             ax.scatter(model[:, 0], model[:, 1], model[:, 2], color='grey', label='Model Data')

#         ax.set_title('3D Scatter plot of sliced clusters and model data with best points')
#         ax.legend()
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#         ax.set_aspect('equal', 'box')  # Set equal scaling using the 'box' aspect
#         plt.show()

#     return cluster_results