import dask.array as da
import dask.dataframe as dd
import glob
import numpy as np
import os

print('Starting Merge')

# Function to merge and save features incrementally with Dask
def merge_and_save_features_dask(prefix, chunk_size=5, start_chunk=4):  # Set chunk size to 5
    # Find all files matching the pattern for the given prefix
    feature_files = sorted(glob.glob(f'./{prefix}_features_batch_*.npy'))
    label_files = sorted(glob.glob(f'./{prefix}_labels_batch_*.csv'))

    # Initialize merged Dask arrays and dataframes with existing files
    merged_features = np.load(f'./final_{prefix}_features.npy') if os.path.exists(f'./final_{prefix}_features.npy') else None
    merged_labels = dd.read_csv(f'./final_{prefix}_labels.csv') if os.path.exists(f'./final_{prefix}_labels.csv') else None

    # Process files in chunks, skipping already processed chunks
    for i in range(start_chunk * chunk_size, len(feature_files), chunk_size):
        # Initialize an empty list to hold chunked Dask arrays
        chunk_features = []

        # Load a chunk of feature files using Dask arrays
        for feature_file in feature_files[i:i + chunk_size]:
            # Load numpy arrays and ensure they have a consistent dtype (e.g., float32)
            array = np.load(feature_file, allow_pickle=True)
            
            if array.dtype == 'object':
                # Convert to float32 if the dtype is 'object'
                array = np.array(array.tolist(), dtype=np.float32)
            
            # Determine appropriate chunks
            array_shape = array.shape
            chunk_size_for_array = (1,) + array_shape[1:]  # Keeping the first dimension small for chunking
            
            # Use Dask to handle chunked arrays
            dask_array = da.from_array(array, chunks=chunk_size_for_array)  # Adjust chunks as needed
            chunk_features.append(dask_array)

        # Merge the chunk features using Dask
        chunk_features_merged = da.concatenate(chunk_features, axis=0)

        # Append to the merged features incrementally
        if merged_features is None:
            merged_features = chunk_features_merged
        else:
            merged_features = da.concatenate([da.from_array(merged_features), chunk_features_merged], axis=0)

        # Load and concatenate label files using Dask DataFrames
        chunk_labels = [dd.read_csv(label_file) for label_file in label_files[i:i + chunk_size]]
        chunk_labels_merged = dd.concat(chunk_labels)

        # Append to the merged labels incrementally
        if merged_labels is None:
            merged_labels = chunk_labels_merged
        else:
            merged_labels = dd.concat([merged_labels, chunk_labels_merged])

        # Print progress
        print(f"Processed chunk {i // chunk_size + 1} / {len(feature_files) // chunk_size + 1}")

        # Save merged features and labels incrementally
        try:
            # Force computation and save merged features to disk
            print("Computing and saving merged features incrementally...")
            merged_features = merged_features.compute()  # Compute the Dask array
            np.save(f'./final_{prefix}_features.npy', merged_features)  # Save as a single NumPy file
            print(f"Feature file './final_{prefix}_features.npy' saved after processing chunk {i // chunk_size + 1}.")
        except Exception as e:
            print(f"Error saving merged features after chunk {i // chunk_size + 1}: {e}")

        try:
            # Force computation and save merged labels to CSV
            print("Computing and saving merged labels incrementally...")
            merged_labels = merged_labels.compute()  # Compute the Dask DataFrame
            merged_labels.to_csv(f'./final_{prefix}_labels.csv', index=False)
            print(f"Label file './final_{prefix}_labels.csv' saved after processing chunk {i // chunk_size + 1}.")
        except Exception as e:
            print(f"Error saving merged labels after chunk {i // chunk_size + 1}: {e}")

    print(f"All {prefix} data batches merged and saved.")

# Merge and save remaining training data incrementally with Dask
merge_and_save_features_dask('train', chunk_size=5, start_chunk=4)

print("Remaining training data batches merged and saved as final set using Dask.")
