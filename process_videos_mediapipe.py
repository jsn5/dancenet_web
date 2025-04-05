import os
import json
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs("data/videos", exist_ok=True)
    os.makedirs("data/processed_poses", exist_ok=True)

def extract_poses_with_mediapipe(video_folder, output_folder):
    """
    Extract poses from videos using MediaPipe Pose.
    
    Args:
        video_folder (str): Path to folder containing input videos.
        output_folder (str): Path to save extracted pose data.
    """
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
    except ImportError:
        print("MediaPipe not found. Installing...")
        import subprocess
        subprocess.call(['pip', 'install', 'mediapipe'])
        import mediapipe as mp
        mp_pose = mp.solutions.pose
    
    # Get list of video files
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print(f"No video files found in {video_folder}. Make sure to add some videos first!")
        return
        
    print(f"Found {len(video_files)} videos to process")
    
    # Initialize MediaPipe Pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1  # 0=Lite, 1=Full, 2=Heavy
    )
    
    # Process each video
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        output_path = os.path.join(output_folder, f"{video_name}_poses.json")
        
        # Skip if already processed
        if os.path.exists(output_path):
            print(f"Skipping {video_file} (already processed)")
            continue
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            continue
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process video frames
        pose_results = []
        frame_id = 0
        
        # Set the number of frames to process (limit for large videos)
        max_frames = total_frames #min(total_frames, 1000)  # Process up to 1000 frames
        
        with tqdm(total=max_frames, desc=f"Processing {video_file} frames", leave=False) as pbar:
            while cap.isOpened() and frame_id < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames for efficiency (process every 2nd frame)
                #if frame_id % 2 != 0:
                #    frame_id += 1
                #    pbar.update(1)
                #    continue
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame with MediaPipe
                results = pose.process(rgb_frame)
                
                # Check if pose detected
                if results.pose_landmarks:
                    # Extract keypoints and scores
                    keypoints = []
                    scores = []
                    
                    for landmark in results.pose_landmarks.landmark:
                        # Get image dimensions
                        h, w, _ = frame.shape
                        
                        # Convert normalized coordinates to pixel coordinates
                        px = min(int(landmark.x * w), w - 1)
                        py = min(int(landmark.y * h), h - 1)
                        
                        keypoints.append([float(px), float(py)])
                        scores.append(float(landmark.visibility))
                    
                    # Add to results
                    pose_results.append({
                        'frame_id': frame_id,
                        'keypoints': keypoints,
                        'scores': scores
                    })
                
                # Increment frame counter
                frame_id += 1
                pbar.update(1)
        
        # Release video capture
        cap.release()
        
        # Skip if no poses detected
        if not pose_results:
            print(f"No valid poses detected in {video_file}. Skipping.")
            continue
            
        # Save results to JSON
        with open(output_path, 'w') as f:
            json.dump({
                'video_name': video_name,
                'fps': float(fps),
                'total_frames': total_frames,
                'num_processed_frames': len(pose_results),
                'poses': pose_results
            }, f, indent=2)
        
        print(f"Saved pose data for {video_file} ({len(pose_results)} frames)")
    
    # Release resources
    pose.close()

def main():
    parser = argparse.ArgumentParser(description="Extract poses from dance videos using MediaPipe")
    parser.add_argument('--video_folder', type=str, default='data/videos', help='Folder containing input videos')
    parser.add_argument('--output_folder', type=str, default='data/processed_poses', help='Folder to save processed pose data')
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Extract poses
    extract_poses_with_mediapipe(args.video_folder, args.output_folder)

if __name__ == "__main__":
    main()
