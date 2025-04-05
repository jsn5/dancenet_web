"""
Create a seed sequence JSON file from raw pose data for model generation.
This script extracts a sequence of poses from a pose JSON file and creates
a seed sequence that can be used for initiating the dance generation process.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from normalization import compute_normalization_params, normalize_keypoints

def load_pose_data(pose_file):
    """
    Load pose data from a JSON file.
    
    Args:
        pose_file (str): Path to the pose JSON file.
        
    Returns:
        list: List of poses, where each pose is a list of keypoints.
    """
    try:
        with open(pose_file, 'r') as f:
            data = json.load(f)
        
        if 'poses' in data:
            return data['poses']
        else:
            print(f"Warning: No 'poses' key found in {pose_file}")
            return []
    except Exception as e:
        print(f"Error loading pose file {pose_file}: {e}")
        return []

def filter_poses_by_confidence(poses, confidence_threshold=0.5):
    """
    Filter poses by confidence score.
    
    Args:
        poses (list): List of pose dictionaries.
        confidence_threshold (float): Minimum average confidence score.
        
    Returns:
        list: Filtered poses.
    """
    filtered_poses = []
    
    for pose in poses:
        if 'scores' in pose:
            # Calculate average upper body confidence (typically more reliable)
            upper_body_indices = list(range(0, 17))
            upper_scores = [pose['scores'][i] for i in upper_body_indices if i < len(pose['scores'])]
            
            if upper_scores and np.mean(upper_scores) >= confidence_threshold:
                filtered_poses.append(pose)
    
    return filtered_poses

def extract_seed_sequence(poses, start_idx=0, sequence_length=30, normalize=True):
    """
    Extract a seed sequence from poses.
    
    Args:
        poses (list): List of pose dictionaries.
        start_idx (int): Starting index in the poses list.
        sequence_length (int): Length of the sequence to extract.
        normalize (bool): Whether to normalize the poses.
        
    Returns:
        dict: Seed sequence data.
    """
    if start_idx + sequence_length > len(poses):
        print(f"Warning: Not enough poses (have {len(poses)}, need {start_idx + sequence_length})")
        sequence_length = len(poses) - start_idx
    
    seed_poses = poses[start_idx:start_idx + sequence_length]
    
    # Extract keypoints
    keypoints_list = [pose['keypoints'] for pose in seed_poses]
    
    if normalize:
        # Compute normalization parameters from all keypoints
        all_keypoints = []
        for keypoints in keypoints_list:
            all_keypoints.extend(keypoints)
        
        # Compute normalization parameters
        center, scale = compute_normalization_params(all_keypoints)
        print(f"Normalization parameters: center={center}, scale={scale}")
        
        # Normalize keypoints
        normalized_keypoints = []
        for keypoints in keypoints_list:
            normalized = normalize_keypoints(keypoints, center, scale)
            # Ensure it's flattened
            if normalized.ndim == 2:
                normalized = normalized.flatten()
            normalized_keypoints.append(normalized.tolist())
    else:
        # Just flatten the keypoints
        normalized_keypoints = []
        for keypoints in keypoints_list:
            flattened = [coord for point in keypoints for coord in point]
            normalized_keypoints.append(flattened)
        center = None
        scale = None
    
    # Create seed sequence data
    seed_data = {
        "sequence_length": len(normalized_keypoints),
        "original_file": os.path.basename(str(Path(poses[0].get('source_file', 'unknown')))),
        "start_idx": start_idx,
        "sequence": normalized_keypoints
    }
    
    if normalize:
        seed_data["normalization"] = {
            "center": center.tolist(),
            "scale": float(scale)
        }
    
    return seed_data

def save_seed_json(seed_data, output_path):
    """
    Save seed sequence data to a JSON file.
    
    Args:
        seed_data (dict): Seed sequence data.
        output_path (str): Path to save the JSON file.
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(seed_data, f, indent=2)
        print(f"Seed sequence saved to {output_path}")
    except Exception as e:
        print(f"Error saving seed sequence: {e}")

def main():
    parser = argparse.ArgumentParser(description="Create a seed sequence JSON file from raw pose data")
    
    # Input and output paths
    parser.add_argument('--pose_file', type=str, required=True,
                        help='Path to the pose JSON file')
    parser.add_argument('--output_path', type=str, default='seed_sequence.json',
                        help='Path to save the seed sequence JSON file')
    
    # Sequence parameters
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting index in the pose file')
    parser.add_argument('--sequence_length', type=int, default=30,
                        help='Length of seed sequence')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Minimum average confidence score for filtering poses')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Do not normalize poses (just flatten them)')
    
    args = parser.parse_args()
    
    # Check if pose file exists
    if not os.path.exists(args.pose_file):
        print(f"Error: Pose file not found at {args.pose_file}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pose data
    print(f"Loading poses from {args.pose_file}...")
    poses = load_pose_data(args.pose_file)
    print(f"Loaded {len(poses)} poses")
    
    # Filter poses by confidence
    if args.confidence_threshold > 0:
        print(f"Filtering poses with confidence threshold {args.confidence_threshold}...")
        poses = filter_poses_by_confidence(poses, args.confidence_threshold)
        print(f"Remaining poses after filtering: {len(poses)}")
    
    if len(poses) < args.sequence_length:
        print(f"Warning: Not enough poses after filtering (have {len(poses)}, need {args.sequence_length})")
        if len(poses) == 0:
            print("Error: No valid poses. Try lowering the confidence threshold.")
            return
    
    # Extract seed sequence
    print(f"Extracting seed sequence of length {args.sequence_length} starting at index {args.start_idx}...")
    seed_data = extract_seed_sequence(
        poses, 
        start_idx=args.start_idx, 
        sequence_length=args.sequence_length,
        normalize=not args.no_normalize
    )
    
    # Save seed sequence
    save_seed_json(seed_data, args.output_path)
    
    print("Done!")

if __name__ == "__main__":
    main()
