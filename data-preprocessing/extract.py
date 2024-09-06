from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
import numpy as np
import os

# Load the ResNet50 model pre-trained on ImageNet without the top classification layer
base_model = ResNet50(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(processed_frame_dir, feature_dir):
    """Extract features for each preprocessed frame and save them."""
    os.makedirs(feature_dir, exist_ok=True)
    processed_frames = [f for f in os.listdir(processed_frame_dir) if f.endswith('.npy')]
    processed_frames.sort()

    for frame_file in processed_frames:
        frame_path = os.path.join(processed_frame_dir, frame_file)
        processed_frame = np.load(frame_path)
        
        # Predict the features
        features = model.predict(processed_frame)
        
        # Save features
        feature_path = os.path.join(feature_dir, f"{frame_file.split('.')[0]}_features.npy")
        np.save(feature_path, features)
