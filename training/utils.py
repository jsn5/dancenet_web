import numpy as np
import json
import os
from tqdm import tqdm

# Import unified normalization functions
from normalization import (
    compute_normalization_params,
    normalize_keypoints,
    denormalize_keypoints
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
            # Calculate average confidence score
            avg_score = np.mean(frame['scores'])
            
            # Keep frame if average score is above threshold
            if avg_score >= threshold:
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

def normalize_poses(pose_data, keypoint_indices=None):
    """
    Normalize poses by centering and scaling.
    
    Args:
        pose_data (list): List of pose data dictionaries.
        keypoint_indices (list, optional): Indices of keypoints to keep.
            If None, keep all keypoints.
        
    Returns:
        list: Normalized pose data.
    """
    normalized_data = []
    
    for video_data in tqdm(pose_data, desc="Normalizing poses"):
        normalized_poses = []
        
        for frame in video_data['poses']:
            keypoints = np.array(frame['keypoints'])
            scores = np.array(frame['scores'])
            
            # Filter keypoints if indices provided
            if keypoint_indices is not None:
                filtered_keypoints = keypoints[keypoint_indices]
                filtered_scores = scores[keypoint_indices]
            else:
                filtered_keypoints = keypoints
                filtered_scores = scores
            
            # Compute normalization parameters using unified approach
            center, scale = compute_normalization_params(filtered_keypoints, filtered_scores)
            
            # Normalize keypoints
            if keypoint_indices is not None:
                # If we're only keeping specific keypoints, normalize only those
                normalized = normalize_keypoints(filtered_keypoints, center, scale)
            else:
                # Otherwise normalize all keypoints
                normalized = normalize_keypoints(keypoints, center, scale)
            
            # Flatten keypoints to [x1, y1, x2, y2, ...] format
            if normalized.ndim == 2:
                flattened_keypoints = normalized.flatten().tolist()
            else:
                flattened_keypoints = normalized.tolist()
            
            # Create normalized frame
            normalized_frame = {
                'frame_id': frame['frame_id'],
                'keypoints_norm': flattened_keypoints,
                'scores': filtered_scores.tolist() if keypoint_indices is not None else scores.tolist(),
                'center': center.tolist(),
                'scale': float(scale),
                'interpolated': frame.get('interpolated', False)
            }
            
            normalized_poses.append(normalized_frame)
        
        # Create a new data dictionary with normalized poses
        norm_video_data = video_data.copy()
        norm_video_data['poses'] = normalized_poses
        
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
        tuple: (input_sequences, target_sequences)
    """
    input_sequences = []
    target_sequences = []
    
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
            
            input_sequences.append(input_seq)
            target_sequences.append(target_seq)
    
    return np.array(input_sequences), np.array(target_sequences)

def save_dataset(input_sequences, target_sequences, output_path):
    """
    Save the processed sequences dataset.
    
    Args:
        input_sequences (np.ndarray): Input sequences.
        target_sequences (np.ndarray): Target sequences.
        output_path (str): Path to save the dataset.
    """
    np.savez(
        output_path,
        input_sequences=input_sequences,
        target_sequences=target_sequences
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
    
    # Filter low confidence poses
    print("Filtering low confidence poses...")
    filtered_data = filter_low_confidence_poses(pose_data, threshold=confidence_threshold)
    
    # Interpolate missing frames
    print("Interpolating missing frames...")
    interpolated_data = interpolate_missing_frames(filtered_data, max_gap=max_gap)
    
    # Normalize poses
    print("Normalizing poses...")
    normalized_data = normalize_poses(interpolated_data, keypoint_indices=keypoint_indices)
    
    # Create sequences
    print("Creating sequences...")
    input_sequences, target_sequences = create_sequences(
        normalized_data, sequence_length=sequence_length, step=step
    )
    
    # Save dataset
    print("Saving dataset...")
    save_dataset(input_sequences, target_sequences, output_path)
    
    print(f"Dataset saved to {output_path}")
    print(f"Dataset shape: {input_sequences.shape}, {target_sequences.shape}")
