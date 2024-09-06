import os
from normalize import preprocess_stored_frames
from extract import extract_features

def main():
    # Base directory
    base_dir = os.path.join('..', 'DAiSEE')

    # Initiating Folder Variables
    frame_dir = os.path.join(base_dir, 'DataSet')
    processed_frame_dir = os.path.join(base_dir, 'Processed_Frames')
    feature_dir = os.path.join(base_dir, 'Features')

    print(f"Frame directory: {frame_dir}")
    print(f"Processed frame directory: {processed_frame_dir}")
    print(f"Feature directory: {feature_dir}")


    # Ensure necessary directories exist
    os.makedirs(processed_frame_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)

    # Run preprocessing
    preprocess_stored_frames(frame_dir, processed_frame_dir)

    # Run feature extraction
    extract_features(processed_frame_dir, feature_dir)

if __name__ == "__main__":
    main()
