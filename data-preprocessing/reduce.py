import os
import numpy as np

def reduce_frames(src_directory, keep_every=3):
    """
    Reduces the number of frames in each video directory by keeping only every `keep_every` frame.
    
    Parameters:
    - src_directory: str, path to the source directory containing video folders with processed frames.
    - keep_every: int, keep every nth frame and delete the rest.
    """
    
    for root, _, files in os.walk(src_directory):  # Recursively traverse all subdirectories
        npy_files = sorted([f for f in files if f.endswith('.npy')], key=lambda x: int(x.split('.')[0]))  # Sort .npy files numerically
        
        for idx, file in enumerate(npy_files):
            file_path = os.path.join(root, file)
            if idx % keep_every != 0:  # Keep every 'keep_every' frame; delete others
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            else:
                print(f"Kept: {file_path}")

def main():
    reduce_frames('C:/Users/ahmad/Desktop/EngagementML/DAiSEE/Processed_Frames')

if __name__ == "__main__":
    main()
