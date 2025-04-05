import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
from tqdm import tqdm

# Import unified normalization functions
from normalization import denormalize_keypoints

def load_dataset(dataset_path):
    """
    Load the processed dataset from .npz file.
    
    Args:
        dataset_path (str): Path to the dataset NPZ file.
        
    Returns:
        dict: Dictionary containing the dataset.
    """
    try:
        data = np.load(dataset_path)
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def draw_pose(ax, keypoints, connections=None, keypoint_size=20, line_width=2):
    """
    Draw a pose on a matplotlib axis.
    
    Args:
        ax (matplotlib.axes.Axes): Matplotlib axis.
        keypoints (numpy.ndarray): Array of keypoint coordinates of shape [num_keypoints, 2].
        connections (list, optional): List of keypoint index pairs to draw connections.
        keypoint_size (int): Size of keypoint markers.
        line_width (int): Width of connection lines.
    """
    # Plot keypoints
    ax.scatter(keypoints[:, 0], keypoints[:, 1], c='r', s=keypoint_size)
    
    # Draw connections between keypoints
    if connections is not None:
        for conn in connections:
            start_idx, end_idx = conn
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                ax.plot([start_point[0], end_point[0]], 
                        [start_point[1], end_point[1]], 
                        'b-', linewidth=line_width)

def setup_axes(fig, ax, data, padding=50):
    """
    Setup the axes for visualization with proper limits.
    
    Args:
        fig (matplotlib.figure.Figure): Matplotlib figure.
        ax (matplotlib.axes.Axes): Matplotlib axis.
        data (numpy.ndarray): Array of all keypoints to determine limits.
        padding (int): Padding around the limits.
    """
    # Find valid keypoints (non-zero)
    valid = (data[:, :, 0] != 0) & (data[:, :, 1] != 0)
    valid_keypoints = data[valid]
    
    if len(valid_keypoints) > 0:
        min_x, max_x = np.min(valid_keypoints[:, 0]), np.max(valid_keypoints[:, 0])
        min_y, max_y = np.min(valid_keypoints[:, 1]), np.max(valid_keypoints[:, 1])
        
        # Add padding
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding
        
        # Set axis limits
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(max_y, min_y)  # Invert y-axis for proper orientation
    else:
        # Default limits if no valid points
        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)  # Invert y-axis
    
    # Set title and remove axes
    ax.set_title("Processed Dance Dataset Visualization")
    ax.axis('off')
    
    # Set figure background color
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

def visualize_dataset_as_mp4(dataset_path, output_path, fps=30, sample_interval=1, max_sequences=None, sequences_per_row=3):
    """
    Visualize the processed dataset as an MP4 video.
    
    Args:
        dataset_path (str): Path to the dataset NPZ file.
        output_path (str): Path to save the MP4 file.
        fps (int): Frames per second for the output video.
        sample_interval (int): Interval between samples to visualize.
        max_sequences (int, optional): Maximum number of sequences to visualize.
        sequences_per_row (int): Number of sequences to display in each row.
    """
    # Load dataset
    data = load_dataset(dataset_path)
    if data is None:
        return
    
    # Extract data
    input_sequences = data['input_sequences']
    target_sequences = data['target_sequences']
    centers = data['centers']
    scales = data['scales']
    
    # Determine how many sequences to visualize
    num_sequences = len(input_sequences)
    if max_sequences is not None:
        num_sequences = min(num_sequences, max_sequences)
    
    print(f"Total sequences in dataset: {len(input_sequences)}")
    print(f"Visualizing {num_sequences} sequences")
    
    # Determine grid size for visualization
    sequences_per_col = (num_sequences + sequences_per_row - 1) // sequences_per_row
    
    # MediaPipe pose connections for visualization
    # Adjust based on your keypoint format
    connections = [
        # Face connections
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        # Upper body connections
        (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        # Lower body connections
        (11, 23), (12, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)
    ]
    
    # Calculate the dimensions of each sequence's keypoints
    # Assuming all sequences have the same structure
    sequence_shape = input_sequences[0].shape
    num_keypoints = sequence_shape[1] // 2
    
    # Create a temporary directory to save frames
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create figure and axes for the grid
    fig, axes = plt.subplots(sequences_per_col, sequences_per_row, figsize=(15, 10))
    if sequences_per_col == 1 and sequences_per_row == 1:
        axes = np.array([[axes]])
    elif sequences_per_col == 1:
        axes = axes.reshape(1, -1)
    elif sequences_per_row == 1:
        axes = axes.reshape(-1, 1)
    
    # Hide unused axes
    for i in range(sequences_per_col):
        for j in range(sequences_per_row):
            idx = i * sequences_per_row + j
            if idx >= num_sequences:
                axes[i, j].axis('off')
    
    # Process each frame of the sequences
    sequence_length = sequence_shape[0]
    
    # Calculate shared axis limits
    all_keypoints = []
    
    for seq_idx in range(num_sequences):
        for frame_idx in range(0, sequence_length, sample_interval):
            # Reshape keypoints from [x1, y1, x2, y2, ...] to [[x1, y1], [x2, y2], ...]
            keypoints_flat = input_sequences[seq_idx][frame_idx]
            keypoints = keypoints_flat.reshape(-1, 2)
            
            # Denormalize keypoints using our unified function
            center = centers[seq_idx]
            scale = scales[seq_idx]
            denormalized = denormalize_keypoints(keypoints, center, scale)
            
            all_keypoints.append(denormalized)
    
    all_keypoints = np.vstack(all_keypoints)
    
    # Create video frames
    print("Creating video frames...")
    for frame_idx in tqdm(range(0, sequence_length, sample_interval)):
        # Clear all axes
        for i in range(sequences_per_col):
            for j in range(sequences_per_row):
                axes[i, j].clear()
                axes[i, j].axis('off')
        
        # Draw each sequence
        for seq_idx in range(num_sequences):
            row = seq_idx // sequences_per_row
            col = seq_idx % sequences_per_row
            
            # Reshape keypoints from [x1, y1, x2, y2, ...] to [[x1, y1], [x2, y2], ...]
            keypoints_flat = input_sequences[seq_idx][frame_idx]
            keypoints = keypoints_flat.reshape(-1, 2)
            
            # Denormalize keypoints using our unified function
            center = centers[seq_idx]
            scale = scales[seq_idx]
            denormalized = denormalize_keypoints(keypoints, center, scale)
            
            # Setup axis limits
            setup_axes(fig, axes[row, col], np.array([denormalized]), padding=50)
            
            # Draw pose
            draw_pose(axes[row, col], denormalized, connections=connections)
            
            # Add sequence number
            axes[row, col].text(0.05, 0.95, f"Seq {seq_idx}", transform=axes[row, col].transAxes, 
                              fontsize=10, verticalalignment='top')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save frame
        frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
        plt.savefig(frame_path, dpi=100)
    
    # Create video from frames
    print("Creating video...")
    create_video_from_frames(temp_dir, output_path, fps)
    
    # Clean up
    print("Cleaning up temporary files...")
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)
    
    print(f"Video saved to {output_path}")

def visualize_sequences_as_mp4(dataset_path, output_path, sequence_indices, fps=30):
    """
    Visualize specific sequences from the dataset as an MP4 video.
    
    Args:
        dataset_path (str): Path to the dataset NPZ file.
        output_path (str): Path to save the MP4 file.
        sequence_indices (list): List of sequence indices to visualize.
        fps (int): Frames per second for the output video.
    """
    # Load dataset
    data = load_dataset(dataset_path)
    if data is None:
        return
    
    # Extract data
    input_sequences = data['input_sequences']
    centers = data['centers']
    scales = data['scales']
    
    # Validate sequence indices
    valid_indices = [idx for idx in sequence_indices if idx < len(input_sequences)]
    if not valid_indices:
        print("No valid sequence indices provided.")
        return
    
    print(f"Visualizing sequences: {valid_indices}")
    
    # MediaPipe pose connections for visualization
    connections = [
        # Face connections
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        # Upper body connections
        (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        # Lower body connections
        (11, 23), (12, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)
    ]
    
    # Get sequence length
    sequence_length = input_sequences[0].shape[0]
    
    # Create temporary directory for frames
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Calculate the number of sequences to display
    num_seqs = len(valid_indices)
    fig_width = min(15, num_seqs * 5)  # Adjust width based on number of sequences
    
    # Create figure and axes
    fig, axes = plt.subplots(1, num_seqs, figsize=(fig_width, 8))
    if num_seqs == 1:
        axes = [axes]
    
    # Process each frame
    print("Creating video frames...")
    for frame_idx in tqdm(range(sequence_length)):
        # Clear all axes
        for ax in axes:
            ax.clear()
            ax.axis('off')
        
        # Draw each sequence
        for i, seq_idx in enumerate(valid_indices):
            # Reshape keypoints from [x1, y1, x2, y2, ...] to [[x1, y1], [x2, y2], ...]
            keypoints_flat = input_sequences[seq_idx][frame_idx]
            keypoints = keypoints_flat.reshape(-1, 2)
            
            # Denormalize keypoints using our unified function
            center = centers[seq_idx]
            scale = scales[seq_idx]
            denormalized = denormalize_keypoints(keypoints, center, scale)
            
            # Setup axis
            setup_axes(fig, axes[i], np.array([denormalized]), padding=50)
            
            # Draw pose
            draw_pose(axes[i], denormalized, connections=connections)
            
            # Add sequence number
            axes[i].text(0.05, 0.95, f"Seq {seq_idx}", transform=axes[i].transAxes, 
                       fontsize=10, verticalalignment='top')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save frame
        frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
        plt.savefig(frame_path, dpi=100)
    
    # Create video from frames
    print("Creating video...")
    create_video_from_frames(temp_dir, output_path, fps)
    
    # Clean up
    print("Cleaning up temporary files...")
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)
    
    print(f"Video saved to {output_path}")

def create_video_from_frames(frames_dir, output_path, fps=30):
    """
    Create a video from a sequence of frames.
    
    Args:
        frames_dir (str): Directory containing frame images.
        output_path (str): Path to save the output video.
        fps (int): Frames per second.
    """
    # Get all frame files and sort them
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
    frame_files.sort()
    
    if not frame_files:
        print("No frames found.")
        return
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width, layers = first_frame.shape
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add frames to video
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        video.write(frame)
    
    # Release the video writer
    video.release()

def main():
    parser = argparse.ArgumentParser(description="Visualize processed dance dataset as MP4")
    
    # Input and output paths
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the processed dataset NPZ file')
    parser.add_argument('--output_path', type=str, default='dataset_visualization.mp4',
                        help='Path to save the output MP4 file')
    
    # Visualization options
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for the output video')
    parser.add_argument('--sample_interval', type=int, default=1,
                        help='Interval between frames to sample for visualization')
    parser.add_argument('--max_sequences', type=int, default=9,
                        help='Maximum number of sequences to visualize (use -1 for all)')
    parser.add_argument('--sequences_per_row', type=int, default=3,
                        help='Number of sequences to display in each row')
    
    # Specific sequence visualization
    parser.add_argument('--sequences', type=str, default=None,
                        help='Comma-separated indices of specific sequences to visualize')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset file not found at {args.dataset_path}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    
    # Visualize specific sequences if provided
    if args.sequences:
        sequence_indices = [int(idx) for idx in args.sequences.split(',')]
        visualize_sequences_as_mp4(args.dataset_path, args.output_path, sequence_indices, args.fps)
    else:
        # Visualize dataset with grid layout
        max_sequences = None if args.max_sequences == -1 else args.max_sequences
        visualize_dataset_as_mp4(
            args.dataset_path, 
            args.output_path, 
            fps=args.fps, 
            sample_interval=args.sample_interval,
            max_sequences=max_sequences,
            sequences_per_row=args.sequences_per_row
        )

if __name__ == "__main__":
    main()
