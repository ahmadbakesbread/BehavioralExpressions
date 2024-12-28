from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
import numpy as np
import os

# Load the ResNet50 model pre-trained on ImageNet without the top classification layer
base_model = ResNet50(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(processed_frame_dir, feature_dir):
    """Extract features for each preprocessed frame in all subdirectories and save them."""
    os.makedirs(feature_dir, exist_ok=True)

    # Walk through all subdirectories
    for root, dirs, files in os.walk(processed_frame_dir):
        for frame_file in files:
            if frame_file.endswith('.npy'):
                frame_path = os.path.join(root, frame_file)

                # Create subdirectory in feature_dir to match the structure of processed_frame_dir
                sub_path = os.path.relpath(root, processed_frame_dir)
                feature_dir_sub = os.path.join(feature_dir, sub_path)
                os.makedirs(feature_dir_sub, exist_ok=True)

                # Save features with the same subdirectory structure
                feature_path = os.path.join(feature_dir_sub, f"{frame_file.split('.')[0]}_features.npy")

                # Check if the feature file already exists; skip if it does
                if os.path.exists(feature_path):
                    print(f"Skipping already processed frame: {frame_file}")
                    continue

                # Load processed frame
                processed_frame = np.load(frame_path)

                # Predict the features
                features = model.predict(processed_frame)

                # Save the features
                np.save(feature_path, features)
