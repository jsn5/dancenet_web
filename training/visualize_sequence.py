import os
import sys
import argparse
import torch
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.model import MDNRNN
from normalization import compute_normalization_params, normalize_keypoints, denormalize_keypoints

def load_model(model_path, input_size, hidden_size=128, num_layers=1, num_mixtures=3, 
               output_size=None, rnn_type='lstm'):
    """
    Load the trained MDN-RNN model.
    
    Args:
        model_path (str): Path to the saved model.
        input_size (int): Dimension of input features.
        hidden_size (int): Number of hidden units in the RNN.
        num_layers (int): Number of RNN layers.
        num_mixtures (int): Number of Gaussian mixtures in the MDN.
        output_size (int): Dimension of output features. If None, use input_size.
        rnn_type (str): Type of RNN cell ('lstm' or 'gru').
        
    Returns:
        MDNRNN: The loaded model.
    """
    # If output_size is not specified, use input_size
    if output_size is None:
        output_size = input_size
    
    # Initialize model
    model = MDNRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_mixtures=num_mixtures,
        output_size=output_size,
        rnn_type=rnn_type
    )
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    return model

def load_pose_data(pose_file):
    """
    Load pose data from a JSON file.
    
    Args:
        pose_file (str): Path to the pose JSON file.
        
    Returns:
        list: List of poses, where each pose is a list of keypoints.
    """
    with open(pose_file, 'r') as f:
        data = json.load(f)
    
    return data['poses']

def prepare_seed_sequence(poses, sequence_length=30, start_idx=0, stabilize=True):
    """
    Prepare a seed sequence from saved poses.
    
    Args:
        poses (list): List of poses from the JSON file.
        sequence_length (int): Length of seed sequence.
        start_idx (int): Starting index in the poses list.
        stabilize (bool): Whether to use stabilization for normalization.
        
    Returns:
        tuple: (seed_sequence, original_keypoints, normalization_stats)
    """
    if start_idx + sequence_length > len(poses):
        print(f"Warning: Not enough poses ({len(poses)}) for seed sequence of length {sequence_length} starting at index {start_idx}")
        sequence_length = len(poses) - start_idx
    
    seed_poses = poses[start_idx:start_idx + sequence_length]
    
    # Extract keypoints
    original_keypoints = [pose['keypoints'] for pose in seed_poses]
    
    # Compute normalization statistics
    all_keypoints = []
    for keypoints in original_keypoints:
        all_keypoints.extend(keypoints)
    
    all_keypoints_array = np.array(all_keypoints)
    
    # Use the unified normalization parameter computation
    center, scale = compute_normalization_params(all_keypoints_array)
    
    print(f"Normalization stats: center={center}, scale={scale}")
    
    # Normalize and flatten keypoints
    normalized_sequences = []
    
    for keypoints in original_keypoints:
        # Normalize keypoints
        normalized = normalize_keypoints(keypoints, center, scale)
        
        # Flatten to 1D array if not already
        if normalized.ndim == 2:
            normalized = normalized.flatten()
        
        normalized_sequences.append(normalized.tolist())
    
    # Convert to tensor and add batch dimension
    seed_sequence = torch.tensor(normalized_sequences, dtype=torch.float32).unsqueeze(0)
    
    return seed_sequence, original_keypoints, (center, scale)

def generate_sequence(model, seed_sequence, num_steps=100, temperature=1.0):
    """
    Generate a sequence of poses using the model.
    
    Args:
        model (MDNRNN): The trained model.
        seed_sequence (torch.Tensor): Seed sequence of shape [1, seq_len, feature_dim].
        num_steps (int): Number of steps to generate.
        temperature (float): Temperature for sampling (higher = more random).
        
    Returns:
        torch.Tensor: Generated sequence of shape [1, seq_len + num_steps, feature_dim].
    """
    with torch.no_grad():
        return model.generate_sequence(seed_sequence, num_steps, temperature)

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
    
    # Remove axes
    ax.axis('off')
    
    # Set figure background color
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

def visualize_sequence_as_mp4(original_keypoints, generated_keypoints, output_path, fps=30, connections=None):
    """
    Visualize original and generated sequences side by side as an MP4 video.
    
    Args:
        original_keypoints (list): List of original keypoint sequences.
        generated_keypoints (list): List of generated keypoint sequences.
        output_path (str): Path to save the MP4 file.
        fps (int): Frames per second for the video.
        connections (list, optional): List of keypoint connections for visualization.
    """
    # Create temporary directory for frames
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Set up default connections if not provided
    if connections is None:
        # MediaPipe pose connections
        connections = [
            # Face connections
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            # Upper body connections
            (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            # Lower body connections
            (11, 23), (12, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)
        ]
    
    # Convert original and generated keypoints to numpy arrays
    original_poses = np.array(original_keypoints)
    generated_poses = np.array(generated_keypoints)
    
    # Get max number of frames
    max_frames = max(len(original_poses), len(generated_poses))
    
    # Stack all poses for axis setup
    all_poses = np.vstack([original_poses, generated_poses])
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Set titles
    ax1.set_title("Original Sequence", fontsize=14)
    ax2.set_title("Generated Sequence", fontsize=14)
    
    # Process frames
    print("Creating video frames...")
    for frame_idx in tqdm(range(max_frames)):
        # Clear axes
        ax1.clear()
        ax2.clear()
        
        # Setup axes titles again (they get cleared)
        ax1.set_title("Original Sequence", fontsize=14)
        ax2.set_title("Generated Sequence", fontsize=14)
        
        # Setup axes
        setup_axes(fig, ax1, all_poses, padding=50)
        setup_axes(fig, ax2, all_poses, padding=50)
        
        # Draw original pose if available
        if frame_idx < len(original_poses):
            keypoints = original_poses[frame_idx]
            draw_pose(ax1, keypoints, connections)
        
        # Draw generated pose if available
        if frame_idx < len(generated_poses):
            keypoints = generated_poses[frame_idx]
            draw_pose(ax2, keypoints, connections)
        
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
    ax.scatter(keypoints[:, 0], keypoints[:, 1], c='r', s=keypoint_size, alpha=0.7)
    
    # Draw connections between keypoints
    if connections is not None:
        for conn in connections:
            start_idx, end_idx = conn
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                # Skip if any keypoint is [0, 0] (invisible)
                if np.all(keypoints[start_idx] == 0) or np.all(keypoints[end_idx] == 0):
                    continue
                    
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                ax.plot([start_point[0], end_point[0]], 
                        [start_point[1], end_point[1]], 
                        'b-', linewidth=line_width, alpha=0.8)

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
    parser = argparse.ArgumentParser(description="Visualize model-generated dance sequence as MP4")
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved PyTorch model')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Number of hidden units in the RNN')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of RNN layers')
    parser.add_argument('--num_mixtures', type=int, default=5,
                        help='Number of Gaussian mixtures in the MDN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                        help='Type of RNN cell (lstm or gru)')
                        
    # Generation parameters
    parser.add_argument('--pose_file', type=str, required=True,
                        help='Path to the pose JSON file for seed sequence')
    parser.add_argument('--sequence_length', type=int, default=30,
                        help='Length of seed sequence')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting index in the pose file for seed sequence')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Number of steps to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for sampling (higher = more random)')
                        
    # Output parameters
    parser.add_argument('--output_path', type=str, default='generated_sequence.mp4',
                        help='Path to save the MP4 file')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for the output video')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    # Check if pose file exists
    if not os.path.exists(args.pose_file):
        print(f"Error: Pose file not found at {args.pose_file}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    
    # Load pose data
    poses = load_pose_data(args.pose_file)
    print(f"Loaded {len(poses)} poses from {args.pose_file}")
    
    # Prepare seed sequence
    seed_sequence, original_keypoints, norm_stats = prepare_seed_sequence(
        poses, args.sequence_length, args.start_idx
    )
    center, scale = norm_stats
    input_size = seed_sequence.size(-1)
    print(f"Prepared seed sequence of shape {seed_sequence.shape}")
    
    # Load model
    model = load_model(
        args.model_path,
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_mixtures=args.num_mixtures,
        rnn_type=args.rnn_type
    )
    print(f"Loaded model from {args.model_path}")
    
    # Generate sequence
    print(f"Generating {args.num_steps} steps with temperature {args.temperature}...")
    generated_sequence = generate_sequence(
        model,
        seed_sequence,
        num_steps=args.num_steps,
        temperature=args.temperature
    )
    print(f"Generated sequence of shape {generated_sequence.shape}")
    
    # Convert to numpy and reshape
    generated_features = generated_sequence.squeeze(0).numpy()
    
    # Reshape to [num_frames, num_keypoints, 2]
    num_frames = generated_features.shape[0]
    num_keypoints = generated_features.shape[1] // 2
    
    # Denormalize keypoints
    denormalized_keypoints = []
    for i in range(num_frames):
        # Reshape flat array to [num_keypoints, 2]
        frame_keypoints = generated_features[i].reshape(num_keypoints, 2)
        
        # Denormalize
        denormalized = denormalize_keypoints(frame_keypoints, center, scale)
        denormalized_keypoints.append(denormalized)
    
    # Also denormalize original keypoints for visualization
    denormalized_original = []
    for keypoints in original_keypoints:
        denormalized = np.array(keypoints)  # Original keypoints are already in original coordinate space
        denormalized_original.append(denormalized)
    
    # Extend original keypoints with ground truth from poses if needed
    if len(denormalized_original) < num_frames:
        remaining = num_frames - len(denormalized_original)
        start_idx = args.start_idx + args.sequence_length
        end_idx = min(start_idx + remaining, len(poses))
        
        for i in range(start_idx, end_idx):
            if i < len(poses):
                keypoints = np.array(poses[i]['keypoints'])
                denormalized_original.append(keypoints)
    
    # Visualize and save as MP4
    print("Creating visualization...")
    visualize_sequence_as_mp4(
        denormalized_original,
        denormalized_keypoints,
        args.output_path,
        fps=args.fps
    )

if __name__ == "__main__":
    main()
