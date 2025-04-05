"""
Unified normalization utilities for pose keypoints.
This module provides consistent methods for normalizing and denormalizing pose data
across preprocessing, generation, and visualization modules.
"""

import numpy as np

# MediaPipe pose indices for common body parts
NOSE_IDX = 0
LEFT_EYE_IDX = 2
RIGHT_EYE_IDX = 5
LEFT_EAR_IDX = 7
RIGHT_EAR_IDX = 8
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12
LEFT_ELBOW_IDX = 13
RIGHT_ELBOW_IDX = 14
LEFT_WRIST_IDX = 15
RIGHT_WRIST_IDX = 16
LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24
LEFT_KNEE_IDX = 25
RIGHT_KNEE_IDX = 26
LEFT_ANKLE_IDX = 27
RIGHT_ANKLE_IDX = 28


def compute_normalization_params(keypoints, confidence_scores=None):
    """
    Compute normalization parameters (center and scale) for pose keypoints.
    Uses a hierarchical approach starting with shoulders, then hips, then nose
    with appropriate fallbacks for reliability.
    
    Args:
        keypoints (numpy.ndarray or list): Array of shape [num_keypoints, 2] with keypoint coordinates
                                         or list of [x,y] coordinates
        confidence_scores (numpy.ndarray or list, optional): Confidence scores for keypoints
        
    Returns:
        tuple: (center, scale) for normalization
    """
    # Convert to numpy array if not already
    keypoints_array = np.array(keypoints)
    
    # Filter out zero coordinates which might indicate missing keypoints
    if keypoints_array.size > 0:
        valid_mask = (keypoints_array[:, 0] != 0) & (keypoints_array[:, 1] != 0)
        valid_keypoints = keypoints_array[valid_mask]
    else:
        valid_keypoints = np.array([])
    
    # If no valid keypoints, return default values
    if len(valid_keypoints) == 0:
        return np.array([0.0, 0.0]), 1.0
    
    # Check for valid shoulder keypoints (preferred method)
    has_valid_shoulders = (
        confidence_scores is None or
        (LEFT_SHOULDER_IDX < len(confidence_scores) and 
         RIGHT_SHOULDER_IDX < len(confidence_scores) and
         confidence_scores[LEFT_SHOULDER_IDX] > 0.5 and 
         confidence_scores[RIGHT_SHOULDER_IDX] > 0.5)
    )
    
    if has_valid_shoulders and LEFT_SHOULDER_IDX < len(keypoints_array) and RIGHT_SHOULDER_IDX < len(keypoints_array):
        # Use shoulders for reference (most stable for upper body movement)
        left_shoulder = keypoints_array[LEFT_SHOULDER_IDX]
        right_shoulder = keypoints_array[RIGHT_SHOULDER_IDX]
        
        # Ensure both shoulders are valid
        if (left_shoulder[0] != 0 and left_shoulder[1] != 0 and 
            right_shoulder[0] != 0 and right_shoulder[1] != 0):
            # Center between shoulders
            center = (left_shoulder + right_shoulder) / 2
            
            # Scale based on shoulder width
            scale = np.linalg.norm(right_shoulder - left_shoulder)
            
            # Add a small constant to prevent division by zero
            scale = max(scale, 1e-5)
            
            return center, scale
    
    # Check for valid hip keypoints (alternative for full body)
    has_valid_hips = (
        confidence_scores is None or
        (LEFT_HIP_IDX < len(confidence_scores) and 
         RIGHT_HIP_IDX < len(confidence_scores) and
         confidence_scores[LEFT_HIP_IDX] > 0.5 and 
         confidence_scores[RIGHT_HIP_IDX] > 0.5)
    )
    
    if has_valid_hips and LEFT_HIP_IDX < len(keypoints_array) and RIGHT_HIP_IDX < len(keypoints_array):
        # Use hips for reference (good for full body movement)
        left_hip = keypoints_array[LEFT_HIP_IDX]
        right_hip = keypoints_array[RIGHT_HIP_IDX]
        
        # Ensure both hips are valid
        if (left_hip[0] != 0 and left_hip[1] != 0 and 
            right_hip[0] != 0 and right_hip[1] != 0):
            # Center between hips
            center = (left_hip + right_hip) / 2
            
            # For scale, try to use shoulders if available, otherwise hip width
            if (LEFT_SHOULDER_IDX < len(keypoints_array) and 
                RIGHT_SHOULDER_IDX < len(keypoints_array)):
                left_shoulder = keypoints_array[LEFT_SHOULDER_IDX]
                right_shoulder = keypoints_array[RIGHT_SHOULDER_IDX]
                
                if (left_shoulder[0] != 0 and left_shoulder[1] != 0 and 
                    right_shoulder[0] != 0 and right_shoulder[1] != 0):
                    scale = np.linalg.norm(right_shoulder - left_shoulder)
                else:
                    scale = np.linalg.norm(right_hip - left_hip)
            else:
                scale = np.linalg.norm(right_hip - left_hip)
            
            # Add a small constant to prevent division by zero
            scale = max(scale, 1e-5)
            
            return center, scale
    
    # Fallback to head/nose if other keypoints aren't reliable
    if (confidence_scores is None or 
        (NOSE_IDX < len(confidence_scores) and confidence_scores[NOSE_IDX] > 0.5)):
        
        if NOSE_IDX < len(keypoints_array) and keypoints_array[NOSE_IDX][0] != 0 and keypoints_array[NOSE_IDX][1] != 0:
            # Use nose as center
            center = keypoints_array[NOSE_IDX]
            
            # Try to use ears for scale if available
            has_valid_ears = (
                confidence_scores is None or
                (LEFT_EAR_IDX < len(confidence_scores) and
                 RIGHT_EAR_IDX < len(confidence_scores) and
                 confidence_scores[LEFT_EAR_IDX] > 0.5 and 
                 confidence_scores[RIGHT_EAR_IDX] > 0.5)
            )
            
            if (has_valid_ears and 
                LEFT_EAR_IDX < len(keypoints_array) and 
                RIGHT_EAR_IDX < len(keypoints_array)):
                
                left_ear = keypoints_array[LEFT_EAR_IDX]
                right_ear = keypoints_array[RIGHT_EAR_IDX]
                
                if (left_ear[0] != 0 and left_ear[1] != 0 and 
                    right_ear[0] != 0 and right_ear[1] != 0):
                    # Scale based on distance between ears
                    scale = np.linalg.norm(right_ear - left_ear)
                    scale = max(scale, 1e-5)
                    return center, scale
            
            # If ears not available, use a default scale based on head size
            # or calculate from bounding box
            min_coords = np.min(valid_keypoints, axis=0)
            max_coords = np.max(valid_keypoints, axis=0)
            diagonal = np.linalg.norm(max_coords - min_coords)
            scale = diagonal / 4  # assuming the head is about 1/4 of the full body
            scale = max(scale, 30.0)  # Minimum scale to avoid too small
            
            return center, scale
    
    # Last resort: use all valid keypoints
    center = np.mean(valid_keypoints, axis=0)
    
    # Use the bounding box diagonal for scale
    min_coords = np.min(valid_keypoints, axis=0)
    max_coords = np.max(valid_keypoints, axis=0)
    scale = np.linalg.norm(max_coords - min_coords)
    
    # Ensure scale is not zero
    scale = max(scale, 1.0)
    
    return center, scale


def normalize_keypoints(keypoints, center, scale):
    """
    Normalize keypoints by centering and scaling.
    
    Args:
        keypoints (numpy.ndarray or list): Keypoint coordinates of shape [num_keypoints, 2]
                                          or flattened [x1, y1, x2, y2, ...]
        center (numpy.ndarray or list): Center coordinates [x, y]
        scale (float): Scale factor
        
    Returns:
        numpy.ndarray: Normalized keypoints in the same format as input
    """
    # Convert to numpy arrays
    keypoints_array = np.array(keypoints)
    center_array = np.array(center)
    
    # Check if keypoints are flattened [x1, y1, x2, y2, ...] or shaped [num_keypoints, 2]
    is_flattened = keypoints_array.ndim == 1
    
    # Reshape flattened keypoints to [num_keypoints, 2] if needed
    if is_flattened:
        keypoints_reshaped = keypoints_array.reshape(-1, 2)
    else:
        keypoints_reshaped = keypoints_array
    
    # Center the keypoints
    centered = keypoints_reshaped - center_array
    
    # Scale the keypoints
    normalized = centered / (scale + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Reshape back to original format if input was flattened
    if is_flattened:
        return normalized.flatten()
    
    return normalized


def denormalize_keypoints(normalized_keypoints, center, scale):
    """
    Denormalize keypoints back to original coordinate space.
    
    Args:
        normalized_keypoints (numpy.ndarray or list): Normalized keypoint coordinates
                                                     of shape [num_keypoints, 2]
                                                     or flattened [x1, y1, x2, y2, ...]
        center (numpy.ndarray or list): Center coordinates [x, y]
        scale (float): Scale factor
        
    Returns:
        numpy.ndarray: Denormalized keypoints in the same format as input
    """
    # Convert to numpy arrays
    normalized_array = np.array(normalized_keypoints)
    center_array = np.array(center)
    
    # Check if keypoints are flattened [x1, y1, x2, y2, ...] or shaped [num_keypoints, 2]
    is_flattened = normalized_array.ndim == 1
    
    # Reshape flattened keypoints to [num_keypoints, 2] if needed
    if is_flattened:
        normalized_reshaped = normalized_array.reshape(-1, 2)
    else:
        normalized_reshaped = normalized_array
    
    # Scale back
    scaled = normalized_reshaped * scale
    
    # Add center back
    denormalized = scaled + center_array
    
    # Reshape back to original format if input was flattened
    if is_flattened:
        return denormalized.flatten()
    
    return denormalized


def freeze_lower_body(current_frame, reference_frame, confidence_threshold=0.3):
    """
    Freezes lower body keypoints when they have low confidence.
    
    Args:
        current_frame (dict): Current frame data with keypoints and scores
        reference_frame (dict): Reference frame with reliable lower body position
        confidence_threshold (float): Threshold below which to freeze keypoints
        
    Returns:
        dict: Updated frame with frozen lower body when appropriate
    """
    # Define lower body keypoint indices (MediaPipe format)
    LOWER_BODY_INDICES = list(range(23, 33))  # Indices for hips, knees, ankles, etc.
    
    # Copy current frame to avoid modifying the original
    updated_frame = current_frame.copy()
    updated_keypoints = np.array(current_frame['keypoints']).copy()
    scores = np.array(current_frame['scores'])
    
    # Check if we have enough keypoints
    if (len(updated_keypoints) < max(LOWER_BODY_INDICES) + 1 or
        len(reference_frame['keypoints']) < max(LOWER_BODY_INDICES) + 1):
        return updated_frame
    
    # Check each lower body keypoint
    for idx in LOWER_BODY_INDICES:
        if idx < len(scores) and scores[idx] < confidence_threshold:
            # Low confidence, replace with reference keypoint
            if idx < len(reference_frame['keypoints']):
                # Get upper body reference points (e.g., hip center)
                HIP_CENTER_INDICES = [LEFT_HIP_IDX, RIGHT_HIP_IDX]  # Left and right hip indices
                
                # Calculate current and reference hip positions
                if all(i < len(updated_keypoints) for i in HIP_CENTER_INDICES) and all(i < len(reference_frame['keypoints']) for i in HIP_CENTER_INDICES):
                    current_hips = np.mean([updated_keypoints[i] for i in HIP_CENTER_INDICES], axis=0)
                    reference_hips = np.mean([np.array(reference_frame['keypoints'][i]) for i in HIP_CENTER_INDICES], axis=0)
                    
                    # Calculate the translation to apply
                    translation = current_hips - reference_hips
                    
                    # Apply the translation to maintain the relative position
                    reference_point = np.array(reference_frame['keypoints'][idx])
                    updated_keypoints[idx] = reference_point + translation
    
    # Update the frame with frozen keypoints
    updated_frame['keypoints'] = updated_keypoints.tolist()
    
    return updated_frame
