import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import open3d as o3d

def animate_tracks(rgb, points, trackgroup=None):
    """Animate 2D tracks overlayed on RGB images."""
    cmap = plt.cm.hsv

    # Generate colors for each track (same color for each point across all frames)
    z_list = np.arange(points.shape[0]) if trackgroup is None else np.array(trackgroup)
    z_list = np.random.permutation(np.max(z_list) + 1)[z_list]
    colors = cmap(z_list / (np.max(z_list) + 1))

    # Set up the plot for the animation
    fig, ax = plt.subplots(figsize=(8, 8))
    image_plot = ax.imshow(rgb[0])  # Display first frame as initial image
    scatter_plot = ax.scatter([], [], s=40, edgecolor='white', marker='o')  # Scatter plot for points
    lines = [ax.plot([], [], color=colors[j], linewidth=2)[0] for j in range(points.shape[0])]  # Line objects for trajectories

    # Store trajectories of points for plotting
    trajectories = {i: [] for i in range(points.shape[0])}

    def update_frame(i):
        # Update the background RGB image
        image_plot.set_array(rgb[i])

        # Get valid 2D points for the current frame
        valid_2d = points[:, i, 0] > 0  # Adjust valid criteria as needed
        valid_points = points[valid_2d, i, :2]

        # Update scatter plot with the current frame's points
        scatter_plot.set_offsets(valid_points)
        scatter_plot.set_color(colors[valid_2d])

        # Update trajectories with the current frame's valid points
        for j in range(points.shape[0]):
            if valid_2d[j]:
                trajectories[j].append(points[j, i, :2])  # Store 2D point

        # Update the lines for each point's trajectory
        for j, line in enumerate(lines):
            if len(trajectories[j]) > 1:
                trajectory = np.array(trajectories[j])
                line.set_data(trajectory[:, 0], trajectory[:, 1])
            else:
                line.set_data([], [])

        return [image_plot, scatter_plot] + lines

    # Create the animation
    anim = animation.FuncAnimation(fig, update_frame, frames=rgb.shape[0], interval=100, blit=True)

    plt.axis('off')  # Turn off the axis
    plt.show()

# Load data from JSON file
with open('data_0.json', 'r') as f:
    data = json.load(f)

# Extract required fields
rgb = np.array(data['video'])  # Assuming 'video' is a key for the RGB data
target_points = np.array(data['target_points'])  # Extract target_points

# Call the animate_tracks function
animate_tracks(rgb, target_points)


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import json
# def animate_tracks(rgb, target_points, target_points_3d):
#     num_frames = rgb.shape[0]
#     num_points = target_points.shape[0]  # Number of points
    
#     # Create a color map
#     cmap = plt.cm.hsv
#     colors = cmap(np.linspace(0, 1, num_points))  # Generate unique colors for each point

#     fig, (ax2d, ax3d) = plt.subplots(1, 2, figsize=(12, 6))
#     ax3d = fig.add_subplot(122, projection='3d')

#     for frame_idx in range(num_frames):
#         frame = rgb[frame_idx]  # Extract the current frame
#         points_2d = target_points[frame_idx]  # 2D points for this frame
#         points_3d = target_points_3d[frame_idx]  # 3D points for this frame

#         valid_2d = target_points[:, frame_idx, 0] > -5  # Adjust valid criteria as needed
#         valid_3d = target_points_3d[:, frame_idx, 0] > -5  # Adjust valid criteria as needed

#         ax2d.imshow(frame)
#         for j in range(num_points):
#             if valid_2d[j]:
#                 ax2d.scatter(target_points[j, frame_idx, 0], target_points[j, frame_idx, 1], 
#                              c=colors[j].reshape(1, -1), s=10)  # Use consistent color for each point

#         ax2d.set_title(f'Frame {frame_idx + 1}')
#         ax2d.legend(['2D Points'])

#         for j in range(num_points):
#             if valid_3d[j]:
#                 ax3d.scatter(target_points_3d[j, frame_idx, 0], target_points_3d[j, frame_idx, 1], 
#                              target_points_3d[j, frame_idx, 2], c=colors[j].reshape(1, -1), s=10)  # Use consistent color for each point

#         ax3d.set_title('3D Points')
#         ax3d.legend()
#         ax3d.set_xlabel('X')
#         ax3d.set_ylabel('Y')
#         ax3d.set_zlabel('Z')

#         plt.pause(0.01)  # Pause to create animation effect
#         ax2d.clear()
#         ax3d.clear()
        
#     plt.show()

# with open('data_0.json', 'r') as f:
#     data = json.load(f)

# # Extract required fields
# rgb = np.array(data['video'])  # Assuming 'video' is a key for the RGB data
# target_points = np.array(data['target_points'])  # 2D points
# target_points3d = np.array(data['target_points_3d'])  # 3D points

# # Call the animate_tracks function
# animate_tracks(rgb, target_points, target_points3d)