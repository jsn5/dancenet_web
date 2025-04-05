import os
import sys
import argparse
import numpy as np
import json
from tqdm import tqdm

# Import unified normalization functions
from normalization import (
    compute_normalization_params,
    normalize_keypoints,
    denormalize_keypoints,
    freeze_lower_body
)

def load_pose_data(pose_folder):
    """
    Load pose data from JSON files.
    
    Args:
        pose_folder (str): Path to folder containing pose JSON files.
        
    Returns:
        list: List of pose data dictionaries.
    """
    pose_files = [f for f in os.listdir(pose_folder) if f.endswith('_poses.json')]
    pose_data = []
    
    for pose_file in tqdm(pose_files, desc="Loading pose data"):
        with open(os.path.join(pose_folder, pose_file), 'r') as f:
            data = json.load(f)
            pose_data.append(data)
            
    return pose_data

def filter_low_confidence_poses(pose_data, threshold=0.3):
    """
    Filter frames with low confidence scores.
    
    Args:
        pose_data (list): List of pose data dictionaries.
        threshold (float): Minimum average confidence score threshold.
        
    Returns:
        list: Filtered pose data.
    """
    filtered_data = []
    
    for video_data in pose_data:
        filtered_poses = []
        
        for frame in video_data['poses']:
            # Calculate average confidence score for upper body only
            upper_body_indices = list(range(0, 17))  # Example: indices 0-16 are upper body
            upper_scores = [frame['scores'][i] for i in upper_body_indices if i < len(frame['scores'])]
            
            if upper_scores:
                avg_upper_score = np.mean(upper_scores)
                
                # Keep frame if average upper body score is above threshold
                if avg_upper_score >= threshold:
                    filtered_poses.append(frame)
        
        # Create a new data dictionary with filtered poses
        filtered_video_data = video_data.copy()
        filtered_video_data['poses'] = filtered_poses
        filtered_video_data['num_processed_frames'] = len(filtered_poses)
        
        filtered_data.append(filtered_video_data)
    
    return filtered_data

def interpolate_missing_frames(pose_data, max_gap=5):
    """
    Interpolate missing frames within reasonable gaps.
    
    Args:
        pose_data (list): List of pose data dictionaries.
        max_gap (int): Maximum gap size to interpolate.
        
    Returns:
        list: Pose data with interpolated frames.
    """
    interpolated_data = []
    
    for video_data in pose_data:
        poses = video_data['poses']
        
        # Sort poses by frame_id
        poses.sort(key=lambda x: x['frame_id'])
        
        # Find frame gaps
        frame_ids = [p['frame_id'] for p in poses]
        
        # Create new poses list with interpolated frames
        new_poses = []
        
        for i in range(len(poses) - 1):
            current_frame = poses[i]
            next_frame = poses[i + 1]
            
            # Add current frame to new poses
            new_poses.append(current_frame)
            
            # Check if there's a gap
            gap = next_frame['frame_id'] - current_frame['frame_id']
            
            # Skip if no gap or gap too large
            if gap <= 1 or gap > max_gap:
                continue
                
            # Interpolate frames
            for j in range(1, gap):
                t = j / gap  # Interpolation factor (0 to 1)
                
                # Interpolate keypoints
                current_keypoints = np.array(current_frame['keypoints'])
                next_keypoints = np.array(next_frame['keypoints'])
                interp_keypoints = current_keypoints + t * (next_keypoints - current_keypoints)
                
                # Interpolate scores
                current_scores = np.array(current_frame['scores'])
                next_scores = np.array(next_frame['scores'])
                interp_scores = current_scores + t * (next_scores - current_scores)
                
                # Create interpolated frame
                interp_frame = {
                    'frame_id': current_frame['frame_id'] + j,
                    'keypoints': interp_keypoints.tolist(),
                    'scores': interp_scores.tolist(),
                    'interpolated': True
                }
                
                new_poses.append(interp_frame)
        
        # Add last frame
        new_poses.append(poses[-1])
        
        # Create a new data dictionary with interpolated poses
        interp_video_data = video_data.copy()
        interp_video_data['poses'] = new_poses
        interp_video_data['num_processed_frames'] = len(new_poses)
        
        interpolated_data.append(interp_video_data)
    
    return interpolated_data

def find_reference_frame(sequence, lower_body_threshold=0.7):
    """
    Find a reference frame with good visibility of the entire body.
    
    Args:
        sequence (list): List of frame dictionaries
        lower_body_threshold (float): Confidence threshold for lower body
        
    Returns:
        dict: Reference frame or None if not found
    """
    # Look for frames with good lower body visibility
    for frame in sequence:
        # Check if frame has scores for lower body keypoints
        if len(frame['scores']) > 30:  # Assuming lower body starts after index 23
            lower_body_scores = frame['scores'][23:30]
            
            if np.mean(lower_body_scores) >= lower_body_threshold:
                return frame
    
    # If no ideal frame found, return the frame with highest average confidence
    if sequence:
        best_frame = max(sequence, key=lambda f: np.mean(f['scores']) if f['scores'] else 0)
        return best_frame
    
    return None

def process_video_sequence(sequence):
    """
    Process a sequence of poses with stabilization and leg freezing.
    
    Args:
        sequence (list): List of frame dictionaries
        
    Returns:
        list: Processed sequence with normalized poses
    """
    # Find a good reference frame with visible lower body
    reference_frame = find_reference_frame(sequence)
    
    processed_frames = []
    
    for frame in sequence:
        # Apply lower body freezing if needed
        if reference_frame:
            stabilized_frame = freeze_lower_body(frame, reference_frame)
        else:
            stabilized_frame = frame
        
        # Get keypoints and scores
        keypoints = np.array(stabilized_frame['keypoints'])
        scores = np.array(stabilized_frame['scores'])
        
        # Compute normalization parameters using unified approach
        center, scale = compute_normalization_params(keypoints, scores)
        
        # Normalize pose
        normalized_keypoints = normalize_keypoints(keypoints, center, scale)
        
        # Flatten keypoints to [x1, y1, x2, y2, ...] format
        if normalized_keypoints.ndim == 2:
            flattened_keypoints = normalized_keypoints.flatten().tolist()
        else:
            flattened_keypoints = normalized_keypoints.tolist()
        
        # Create processed frame
        processed_frame = {
            'frame_id': frame['frame_id'],
            'keypoints_norm': flattened_keypoints,
            'scores': scores.tolist(),
            'center': center.tolist(),
            'scale': float(scale),
            'interpolated': frame.get('interpolated', False)
        }
        
        processed_frames.append(processed_frame)
    
    return processed_frames

def normalize_poses(pose_data, keypoint_indices=None):
    """
    Normalize poses with stabilization and leg freezing.
    
    Args:
        pose_data (list): List of pose data dictionaries.
        keypoint_indices (list, optional): Indices of keypoints to keep.
            If None, keep all keypoints.
        
    Returns:
        list: Normalized pose data.
    """
    normalized_data = []
    
    for video_data in tqdm(pose_data, desc="Normalizing videos"):
        # Process each video sequence
        processed_frames = process_video_sequence(video_data['poses'])
        
        # Filter keypoints if indices provided
        if keypoint_indices is not None:
            for frame in processed_frames:
                # Extract and filter normalized keypoints
                norm_keypoints = []
                for i in range(0, len(frame['keypoints_norm']), 2):
                    if i//2 in keypoint_indices and i+1 < len(frame['keypoints_norm']):
                        norm_keypoints.extend([frame['keypoints_norm'][i], frame['keypoints_norm'][i+1]])
                
                frame['keypoints_norm'] = norm_keypoints
                
                # Filter scores
                if 'scores' in frame and keypoint_indices:
                    frame['scores'] = [frame['scores'][i] for i in keypoint_indices if i < len(frame['scores'])]
        
        # Create a new data dictionary with normalized poses
        norm_video_data = video_data.copy()
        norm_video_data['poses'] = processed_frames
        
        normalized_data.append(norm_video_data)
    
    return normalized_data

def create_sequences(pose_data, sequence_length=30, step=1):
    """
    Create sequences of normalized poses for training.
    
    Args:
        pose_data (list): List of normalized pose data dictionaries.
        sequence_length (int): Length of input sequences.
        step (int): Step size for sliding window.
        
    Returns:
        tuple: (input_sequences, target_sequences, normalization_stats)
    """
    input_sequences = []
    target_sequences = []
    normalization_stats = []
    
    for video_data in pose_data:
        poses = video_data['poses']
        
        # Skip videos with too few frames
        if len(poses) <= sequence_length:
            continue
        
        # Extract normalized keypoints
        keypoints_norm = [frame['keypoints_norm'] for frame in poses]
        
        # Create sequences with sliding window
        for i in range(0, len(keypoints_norm) - sequence_length, step):
            input_seq = keypoints_norm[i:i + sequence_length]
            target_seq = keypoints_norm[i + sequence_length]
            
            # Get normalization stats from the last frame in sequence
            center = poses[i + sequence_length - 1]['center']
            scale = poses[i + sequence_length - 1]['scale']
            
            input_sequences.append(input_seq)
            target_sequences.append(target_seq)
            normalization_stats.append((center, scale))
    
    return np.array(input_sequences), np.array(target_sequences), normalization_stats

def save_dataset(input_sequences, target_sequences, normalization_stats, output_path):
    """
    Save the processed sequences dataset.
    
    Args:
        input_sequences (np.ndarray): Input sequences.
        target_sequences (np.ndarray): Target sequences.
        normalization_stats (list): List of (center, scale) tuples.
        output_path (str): Path to save the dataset.
    """
    # Convert normalization stats to arrays
    centers = [stat[0] for stat in normalization_stats]
    scales = [stat[1] for stat in normalization_stats]
    
    np.savez(
        output_path,
        input_sequences=input_sequences,
        target_sequences=target_sequences,
        centers=centers,
        scales=scales
    )
    
def process_pose_data(pose_folder, output_path, sequence_length=30, step=1, 
                      confidence_threshold=0.3, max_gap=5, keypoint_indices=None):
    """
    End-to-end processing of pose data for training.
    
    Args:
        pose_folder (str): Path to folder containing pose JSON files.
        output_path (str): Path to save the processed dataset.
        sequence_length (int): Length of input sequences.
        step (int): Step size for sliding window.
        confidence_threshold (float): Minimum average confidence score.
        max_gap (int): Maximum gap size to interpolate.
        keypoint_indices (list, optional): Indices of keypoints to keep.
    """
    # Load pose data
    print("Loading pose data...")
    pose_data = load_pose_data(pose_folder)
    
    # Filter low confidence poses (focusing on upper body)
    print("Filtering low confidence poses...")
    filtered_data = filter_low_confidence_poses(pose_data, threshold=confidence_threshold)
    
    # Interpolate missing frames
    print("Interpolating missing frames...")
    interpolated_data = interpolate_missing_frames(filtered_data, max_gap=max_gap)
    
    # Normalize poses with stabilization and leg freezing
    print("Normalizing poses with stabilization...")
    normalized_data = normalize_poses(interpolated_data, keypoint_indices=keypoint_indices)
    
    # Create sequences
    print("Creating sequences...")
    input_sequences, target_sequences, normalization_stats = create_sequences(
        normalized_data, sequence_length=sequence_length, step=step
    )
    
    # Save dataset
    print("Saving dataset...")
    save_dataset(input_sequences, target_sequences, normalization_stats, output_path)
    
    print(f"Dataset saved to {output_path}")
    print(f"Dataset shape: {input_sequences.shape}, {target_sequences.shape}")
    print(f"Number of normalization stats: {len(normalization_stats)}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess pose data with stabilization for training")
    parser.add_argument('--pose_folder', type=str, default='data/processed_poses', help='Folder containing pose JSON files')
    parser.add_argument('--output_path', type=str, default='data/processed_poses/dance_dataset_stabilized.npz', help='Path to save the processed dataset')
    parser.add_argument('--sequence_length', type=int, default=30, help='Length of input sequences')
    parser.add_argument('--step', type=int, default=1, help='Step size for sliding window')
    parser.add_argument('--confidence_threshold', type=float, default=0.3, help='Minimum average confidence score')
    parser.add_argument('--max_gap', type=int, default=5, help='Maximum gap size to interpolate')
    parser.add_argument('--keypoint_indices', type=str, default=None, help='Comma-separated indices of keypoints to keep (e.g., "0,1,2,3,4,5")')
    parser.add_argument('--upper_body_only', action='store_true', help='Use only upper body keypoints')
    
    args = parser.parse_args()
    
    # Parse keypoint indices if provided
    keypoint_indices = None
    if args.keypoint_indices:
        keypoint_indices = [int(idx) for idx in args.keypoint_indices.split(',')]
    elif args.upper_body_only:
        # Use common upper body keypoints (adjust for your keypoint format)
        keypoint_indices = list(range(0, 17))  # Upper body keypoints (0-16)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Process pose data
    process_pose_data(
        args.pose_folder,
        args.output_path,
        sequence_length=args.sequence_length,
        step=args.step,
        confidence_threshold=args.confidence_threshold,
        max_gap=args.max_gap,
        keypoint_indices=keypoint_indices
    )

if __name__ == "__main__":
    main()
