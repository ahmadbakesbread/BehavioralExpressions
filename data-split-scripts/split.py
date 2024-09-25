'''import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import glob
import gc  # Garbage collection to free up memory

# Step 1: Load Labels from CSV
print("Loading labels from CSV...")
labels_df = pd.read_csv('..\DAiSEE\Labels\TestLabels.csv')  # Adjust the path to your labels file
print("Labels loaded successfully.")

# Create a dictionary to map ClipID to its labels
labels_dict = labels_df.set_index('ClipID').T.to_dict('list')

# Step 2: Prepare feature set
features_dir = '..\DAiSEE\Features\Test'  
video_features = {} 
print("Loading feature files...")


# Create directory to save aggregated features if it doesn't exist
aggregated_dir = './aggregated_features'
os.makedirs(aggregated_dir, exist_ok=True)
print(f"Directory '{aggregated_dir}' created or already exists.")

# Function to save aggregated features to disk
def save_video_features(video_folder, features):
    output_path = f'./aggregated_features/{video_folder}_features.npy'
    np.save(output_path, features)
    print(f"Saved features for video {video_folder} to disk.")


# Loop through each person's folder and then through each video's folder
person_count = 0  # Counter for persons
for person_folder in os.listdir(features_dir):
    person_path = os.path.join(features_dir, person_folder)
    if not os.path.isdir(person_path):
        continue

    # Print progress per person folder
    print(f"Processing person {person_count + 1}: {person_folder}")
    person_count += 1

    video_count = 0  # Counter for videos
    for video_folder in os.listdir(person_path):
        video_path = os.path.join(person_path, video_folder)
        if not os.path.isdir(video_path):
            continue

        # Collect all frame features for this video using memory mapping
        frames = glob.glob(os.path.join(video_path, '*.npy'))
        frame_features = []
        for frame in frames:
            try:
                frame_features.append(np.load(frame, mmap_mode='r'))  # Use memory mapping
            except MemoryError as e:
                print(f"MemoryError loading frame: {frame} - {e}")
                continue  # Skip the problematic frame

        # Aggregate features for the video and save to disk
        if frame_features:
            try:
                stacked_features = np.vstack(frame_features)  # Stack all frames vertically
                save_video_features(video_folder, stacked_features)
            except MemoryError as e:
                print(f"MemoryError stacking frames for video {video_folder}: {e}")
                continue  # Skip this video if memory is an issue

        video_count += 1

    print(f"Processed {video_count} videos for person {person_folder}")

    # Clear memory
    gc.collect()

print("All features loaded and saved to disk.")

# Step 3: Prepare Data for Splitting
X = []  # Aggregated features
y = []  # Corresponding labels

print("Mapping features to labels...")
for i, (video_folder, labels) in enumerate(labels_dict.items()):
    feature_path = f'./aggregated_features/{video_folder}_features.npy'
    if os.path.exists(feature_path):
        X.append(np.load(feature_path, allow_pickle=True))
        y.append(labels)

    # Print progress every 100 videos
    if (i + 1) % 100 == 0:
        print(f"Mapped labels to features for {i + 1} videos.")

print("Data preparation complete.")

# Convert to numpy arrays for splitting
X = np.array(X, dtype=object)  # Use dtype=object for varying frame sizes
y = np.array(y)

# Step 4: Split Data into Train, Test, Validation
print("Splitting data into train, test, and validation sets...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y[:, 1])
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42, stratify=y_temp[:, 1])
print("Data splitting complete.")

# Step 5: Save New Splits
def save_features_labels(features, labels, prefix):
    print(f"Saving {prefix} features and labels...")
    np.save(f'./new_{prefix}_features.npy', features)
    pd.DataFrame(labels, columns=['Boredom', 'Engagement', 'Confusion', 'Frustration']).to_csv(f'./new_{prefix}_labels.csv', index=False)
    print(f"{prefix.capitalize()} features and labels saved.")

save_features_labels(X_train, y_train, 'train')
save_features_labels(X_test, y_test, 'test')
save_features_labels(X_val, y_val, 'val')

print("All data saved successfully.")
'''
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import glob

# Set paths
features_dir = 'C:/Users/ahmad/Desktop/EngagementML/DAiSEE/Features/Test'
labels_df = pd.read_csv('C:/Users/ahmad/Desktop/EngagementML/DAiSEE/Labels/TestLabels.csv')
labels_dict = labels_df.set_index('ClipID').T.to_dict('list')

def save_features_labels(features, labels, prefix, batch_num):
    print(f"Saving {prefix} features and labels for batch {batch_num}...")
    np.save(f'./{prefix}_features_batch_{batch_num}.npy', features)
    
    # Convert labels to DataFrame including ClipID
    labels_df = pd.DataFrame(labels, columns=['ClipID', 'Boredom', 'Engagement', 'Confusion', 'Frustration'])
    labels_df.to_csv(f'./{prefix}_labels_batch_{batch_num}.csv', index=False)
    print(f"Saved {prefix} set with {len(features)} samples for batch {batch_num}.")

def load_and_aggregate_features(video_id):
    video_folder = video_id.rsplit('.', 1)[0]
    person_id = video_folder[:6]
    video_path = os.path.join(features_dir, person_id, video_folder)
    print(f"Loading features for video ID: {video_id} from {video_path}")
    
    if os.path.isdir(video_path):
        frame_files = glob.glob(os.path.join(video_path, '*_features.npy'))
        print(f"Found {len(frame_files)} frame files for video ID: {video_id}")
        if frame_files:
            frame_features = [np.load(frame, mmap_mode='r') for frame in frame_files]
            return np.vstack(frame_features)
    print(f"No features found or directory does not exist for video ID: {video_id}")
    return None

def safe_split(X, y, stratify):
    try:
        if len(y) == 0:  # Check if y is empty
            print("Empty batch. Skipping split.")
            return X, [], [], y, [], []

        # Determine if stratification is possible
        unique, counts = np.unique(y[:, 2], return_counts=True)
        stratify_possible = np.all(counts >= 2) and stratify

        if stratify_possible:
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y[:, 2])
        else:
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

        if len(X_temp) > 1:
            if stratify_possible:
                unique, counts = np.unique(y_temp[:, 2], return_counts=True)
                stratify_possible_temp = np.all(counts >= 2)
                if stratify_possible_temp:
                    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42, stratify=y_temp[:, 2])
                else:
                    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)
            else:
                X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)
        else:
            X_test, X_val, y_test, y_val = X_temp, [], y_temp, []

        return X_train, X_test, X_val, y_train, y_test, y_val

    except ValueError as e:
        print(f"Split failed: {e}. Using entire batch as training set.")
        return X, [], [], y, [], []

# Prepare data in batches
batch_size = 30  # Adjusted for testing
X, y = [], []

print("Starting batch processing...")

for i, (video_id, labels) in enumerate(labels_dict.items()):
    print(f"Processing video {i + 1}/{len(labels_dict)}: {video_id}")
    features = load_and_aggregate_features(video_id)
    if features is not None:
        X.append(features)
        # Include ClipID in labels
        y.append([video_id] + labels)
    else:
        print(f"No features loaded for: {video_id}")  # Debugging missing features
        
    if (i + 1) % batch_size == 0 or i == len(labels_dict) - 1:
        print(f"Processing batch {i // batch_size + 1} with {len(X)} samples...")

        if len(X) == 0 or len(y) == 0:  # Skip empty batches
            print("Empty batch. Skipping processing.")
            continue
        
        # Convert to numpy arrays
        X = np.array(X, dtype=object)
        y = np.array(y, dtype=object)  # Include ClipID
        
        # Perform the safe split
        X_train, X_test, X_val, y_train, y_test, y_val = safe_split(X, y, stratify=True)
        
        # Saving the data
        batch_num = i // batch_size + 1  # Batch number for naming
        if len(X_train) > 0:
            save_features_labels(X_train, y_train, 'train', batch_num)
        if len(X_test) > 0:
            save_features_labels(X_test, y_test, 'test', batch_num)
        if len(X_val) > 0:
            save_features_labels(X_val, y_val, 'val', batch_num)
        
        print(f"Batch {batch_num} processed and saved.")
        # Clear batch data to free memory
        X, y = [], []

print("Batch processing complete.")
