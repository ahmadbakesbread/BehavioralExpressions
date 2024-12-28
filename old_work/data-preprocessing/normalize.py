from PIL import Image
import os
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image


def resize_frame(img, size=(224, 224)):
    """Resize a single frame to the specified size."""
    return img.resize(size, Image.Resampling.LANCZOS)  # Updated from Image.ANTIALIAS to Image.Resampling.LANCZOS

def prepare_image(img):
    """Convert and normalize a single frame."""
    img_array = image.img_to_array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    img_array = preprocess_input(img_array)  # Normalize the image
    return img_array

def preprocess_frame(frame_path):
    """Load, resize, and preprocess a frame."""
    img = Image.open(frame_path)  # Load image
    img = resize_frame(img, size=(224, 224))  # Resize image
    img_array = prepare_image(img)  # Preprocess image
    return img_array

def preprocess_stored_frames(frame_dir, processed_frame_dir):
    """Load, preprocess, and save every third frame from all subdirectories."""
    os.makedirs(processed_frame_dir, exist_ok=True)
    processed_count = 0
    skipped_files = 0

    for root, dirs, files in os.walk(frame_dir):
        # Sort files to ensure consistent processing order
        files = sorted([f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        for idx, file in enumerate(files):
            if idx % 3 == 0:  # Process only every third frame
                frame_path = os.path.join(root, file)
                sub_path = os.path.relpath(root, frame_dir)
                processed_frame_dir_sub = os.path.join(processed_frame_dir, sub_path)
                processed_frame_path = os.path.join(processed_frame_dir_sub, file + '.npy')  # Save as .npy

                # Check if already processed
                if os.path.exists(processed_frame_path):
                    skipped_files += 1
                    continue

                # Create directory if it doesn't exist
                os.makedirs(processed_frame_dir_sub, exist_ok=True)

                try:
                    processed_frame = preprocess_frame(frame_path)
                    np.save(processed_frame_path, processed_frame)
                    processed_count += 1
                except Exception:
                    pass  # Silently handle exceptions without printing

    # Commented out summary print statement
    print(f"Total frames processed: {processed_count}, skipped: {skipped_files}")


def test_single_image(frame_path):
    """Process a single image from a specified path and save the output."""
    processed_frame_dir = '..\DAiSEE\Processed_Frames'  # Path to save processed frames
    try:
        processed_frame = preprocess_frame(frame_path)  # Process the image
        processed_frame_path = os.path.join(processed_frame_dir, os.path.basename(frame_path))
        np.save(processed_frame_path, processed_frame)
        print(f"Test image processed and saved to {processed_frame_path}")
    except Exception as e:
        print(f"Error processing image {frame_path}: {e}")

# This test was conducted to verify if a single image can be processed successfully.
if __name__ == "__main__":
    frame_path = '..//DAiSEE//DataSet//Test//500044//5000441001//50004410011.jpg'
    test_single_image(frame_path)
