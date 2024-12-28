from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
import os
import numpy as np


# Directory containing extracted features saved as .npy files
features_dir = '../DAiSEE/Features'

def load_features(directory):
    """
    Load all feature files from the specified directory and stack them vertically to create a matrix.
    
    Parameters:
    - directory: str, path to the directory containing .npy files of extracted features.
    
    Returns:
    - A 2D numpy array where each row is a set of features for a frame.
    """
    
    all_features = []
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):  
            file_path = os.path.join(directory, filename)
            features = np.load(file_path)
            all_features.append(features)
    return np.vstack(all_features)  # Stack features vertically to create a matrix


# Load all features from the specified directory
features = load_features(features_dir)

# Normalize the features to have a mean of 0 and standard deviation of 1
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Apply PCA to reduce dimensionality while retaining 95% of the variance
pca = PCA(n_components=0.95)
features_reduced = pca.fit_transform(features_normalized)