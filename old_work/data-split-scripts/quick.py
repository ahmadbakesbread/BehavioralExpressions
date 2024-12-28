import os
import numpy as np
import glob

def load_and_aggregate_features(base_path, video_id):
    video_folder = video_id.replace('.avi', '')
    target_path = os.path.join(base_path, video_folder)
    target_path = os.path.normpath(target_path)

    print(f"Searching in: {target_path}")

    # Explicitly list all files in the directory for debugging
    all_files = os.listdir(target_path)
    print(f"All files in directory: {all_files}")

    frame_files = glob.glob(os.path.join(target_path, '*_features.npy'))
    print(f"Found {len(frame_files)} frame files using pattern '*_features.npy'.")

    if frame_files:
        frame_features = [np.load(frame, mmap_mode='r') for frame in frame_files]
        if frame_features:
            aggregated_features = np.vstack(frame_features)
            print(f"Aggregated features dimensions: {aggregated_features.shape}")
            return aggregated_features
    else:
        print("No .npy files found using the specified pattern.")
        return None

# Example usage
features_dir = 'C:\\Users\\ahmad\\Desktop\\EngagementML\\DAiSEE\\Features\\Test\\500044'
video_id = '5000441001.avi'
features = load_and_aggregate_features(features_dir, video_id)

