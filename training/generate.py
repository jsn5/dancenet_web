import os
import sys
import argparse
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.model import MDNRNN
from normalization import compute_normalization_params, normalize_keypoints, denormalize_keypoints

def load_model(model_path, input_size, hidden_size=256, num_layers=1, num_mixtures=5, 
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
    
    poses = data['poses']
    return poses

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
    
    # Flatten all keypoints to compute normalization stats
    all_keypoints = []
    for keypoints in original_keypoints:
        all_keypoints.extend(keypoints)
    
    # Compute normalization statistics
    center, scale = compute_normalization_params(all_keypoints)
    print(f"Normalization stats: center={center}, scale={scale}")
    
    # Normalize and flatten keypoints
    normalized_sequences = []
    
    for keypoints in original_keypoints:
        # Normalize keypoints
        normalized = normalize_keypoints(keypoints, center, scale)
        
        # Ensure normalized is flattened
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

def visualize_pose(ax, keypoints, connections=None, keypoint_size=20, line_width=2):
    """
    Visualize a pose on a matplotlib axis.
    
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

def create_animation(original_poses, generated_poses, connections=None, save_path=None, fps=30):
    """
    Create an animation comparing original and generated poses.
    
    Args:
        original_poses (list): List of original pose sequences.
        generated_poses (list): List of generated pose sequences.
        connections (list, optional): List of keypoint index pairs to draw connections.
        save_path (str, optional): Path to save the animation.
        fps (int): Frames per second.
        
    Returns:
        matplotlib.animation.FuncAnimation: The animation object.
    """
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Convert list to numpy arrays
    original_array = np.array(original_poses)
    generated_array = np.array(generated_poses)
    
    # Set axis limits
    all_poses = np.concatenate([original_array.reshape(-1, 2), generated_array.reshape(-1, 2)], axis=0)
    valid_poses = all_poses[(all_poses[:, 0] != 0) & (all_poses[:, 1] != 0)]
    
    if len(valid_poses) > 0:
        min_x, max_x = np.min(valid_poses[:, 0]), np.max(valid_poses[:, 0])
        min_y, max_y = np.min(valid_poses[:, 1]), np.max(valid_poses[:, 1])
        
        # Add some padding
        padding_x = (max_x - min_x) * 0.2
        padding_y = (max_y - min_y) * 0.2
        
        ax1.set_xlim(min_x - padding_x, max_x + padding_x)
        ax1.set_ylim(max_y + padding_y, min_y - padding_y)  # Invert y-axis
        ax2.set_xlim(min_x - padding_x, max_x + padding_x)
        ax2.set_ylim(max_y + padding_y, min_y - padding_y)  # Invert y-axis
    else:
        # Default limits if no valid points
        ax1.set_xlim(-500, 500)
        ax1.set_ylim(500, -500)  # Invert y-axis
        ax2.set_xlim(-500, 500)
        ax2.set_ylim(500, -500)  # Invert y-axis
    
    # Set titles
    ax1.set_title("Original Poses")
    ax2.set_title("Generated Poses")
    
    # Initialize plots (will be updated in animation)
    keypoints1 = ax1.scatter([], [], c='r', s=20)
    keypoints2 = ax2.scatter([], [], c='r', s=20)
    
    # Lists to store line objects
    lines1 = []
    lines2 = []
    
    if connections is not None:
        for _ in connections:
            line1, = ax1.plot([], [], 'b-', linewidth=2)
            line2, = ax2.plot([], [], 'b-', linewidth=2)
            lines1.append(line1)
            lines2.append(line2)
    
    def init():
        # Initialize scatter plots
        keypoints1.set_offsets(np.empty((0, 2)))
        keypoints2.set_offsets(np.empty((0, 2)))
        
        # Initialize lines
        if connections is not None:
            for line in lines1 + lines2:
                line.set_data([], [])
        
        return [keypoints1, keypoints2] + lines1 + lines2
    
    def update(frame):
        # Update keypoints
        if frame < len(original_poses):
            keypoints1.set_offsets(original_poses[frame])
        
        if frame < len(generated_poses):
            keypoints2.set_offsets(generated_poses[frame])
        
        # Update connections
        if connections is not None:
            for i, conn in enumerate(connections):
                start_idx, end_idx = conn
                
                # Original pose connections
                if frame < len(original_poses) and start_idx < len(original_poses[frame]) and end_idx < len(original_poses[frame]):
                    start_point = original_poses[frame][start_idx]
                    end_point = original_poses[frame][end_idx]
                    lines1[i].set_data([start_point[0], end_point[0]], 
                                       [start_point[1], end_point[1]])
                else:
                    lines1[i].set_data([], [])
                    
                # Generated pose connections
                if frame < len(generated_poses) and start_idx < len(generated_poses[frame]) and end_idx < len(generated_poses[frame]):
                    start_point = generated_poses[frame][start_idx]
                    end_point = generated_poses[frame][end_idx]
                    lines2[i].set_data([start_point[0], end_point[0]], 
                                       [start_point[1], end_point[1]])
                else:
                    lines2[i].set_data([], [])
        
        return [keypoints1, keypoints2] + lines1 + lines2
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=max(len(original_poses), len(generated_poses)),
                         init_func=init, blit=True, interval=1000/fps)
    
    # Save animation if path is provided
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Animation saved to {save_path}")
    
    return anim

def main():
    parser = argparse.ArgumentParser(description="Generate dance sequences using trained MDN-RNN model")
    
    # Model parameters
    parser.add_argument('--model_path', type=str, default='training/trained_models/best_model.pth',
                        help='Path to the saved PyTorch model')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Number of hidden units in the RNN')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of RNN layers')
    parser.add_argument('--num_mixtures', type=int, default=6,
                        help='Number of Gaussian mixtures in the MDN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                        help='Type of RNN cell (lstm or gru)')
                        
    # Generation parameters
    parser.add_argument('--pose_file', type=str, required=True,
                        help='Path to the pose JSON file for seed sequence')
    parser.add_argument('--sequence_length', type=int, default=90,
                        help='Length of seed sequence')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting index in the pose file for seed sequence')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Number of steps to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for sampling (higher = more random)')
                        
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='generated',
                        help='Directory to save generated sequences')
    parser.add_argument('--save_animation', action='store_true',
                        help='Save animation of the generated sequence')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for the animation')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please train the model first: python training/train.py")
        return
    
    # Check if pose file exists
    if not os.path.exists(args.pose_file):
        print(f"Error: Pose file not found at {args.pose_file}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load pose data
    poses = load_pose_data(args.pose_file)
    print(f"Loaded {len(poses)} poses from {args.pose_file}")
    
    # Prepare seed sequence and get normalization stats
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
    
    # Convert to list of keypoints for denormalization
    generated_keypoints = []
    
    for i in range(num_frames):
        # Reshape to [num_keypoints, 2]
        frame_keypoints = generated_features[i].reshape(num_keypoints, 2)
        
        # Denormalize
        denormalized = denormalize_keypoints(frame_keypoints, center, scale)
        
        # Add to list
        generated_keypoints.append(denormalized)
    
    # Extract original poses with same number of frames for comparison
    max_frames = len(original_keypoints) + args.num_steps
    max_frames = min(max_frames, len(poses))
    
    all_original_keypoints = [pose['keypoints'] for pose in poses[:max_frames]]
    
    # Define connections for visualization (MediaPipe format)
    # Adjust based on your keypoint format
    connections = [
        # Face connections
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        # Upper body connections
        (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        # Lower body connections
        (11, 23), (12, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)
    ]
    
    # Save generated sequence as JSON
    output_json_path = os.path.join(args.output_dir, "generated_sequence.json")
    with open(output_json_path, 'w') as f:
        json.dump({
            "num_frames": len(generated_keypoints),
            "normalization": {
                "center": center.tolist(),
                "scale": float(scale)
            },
            "poses": [keypoints.tolist() for keypoints in generated_keypoints]
        }, f, indent=2)
    print(f"Saved generated sequence to {output_json_path}")
    
    # Create and save animation
    if args.save_animation:
        animation_path = os.path.join(args.output_dir, "dance_animation.gif")
        
        # Make sure we have enough original poses
        min_frames = min(len(all_original_keypoints), len(generated_keypoints))
        
        anim = create_animation(
            all_original_keypoints[:min_frames],
            generated_keypoints[:min_frames],
            connections=connections,
            save_path=animation_path,
            fps=args.fps
        )
        
        print(f"Animation saved to {animation_path}")
    
    print("Generation complete!")

if __name__ == "__main__":
    main()
