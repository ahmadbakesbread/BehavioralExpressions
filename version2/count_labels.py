import pandas as pd
import matplotlib as plt
import cv2
import os

df = pd.read_csv('C:/Users/ahmad/Desktop/EngagementML/DAiSEE/Labels/AllLabels.csv')

# Initialize maps for counting labels
boredom_map = {'0': 0, '1': 0, '2': 0, '3': 0}
engagement_map = {'0': 0, '1': 0, '2': 0, '3': 0}
confusion_map = {'0': 0, '1': 0, '2': 0, '3': 0}
frustration_map = {'0': 0, '1': 0, '2': 0, '3': 0}

print(df.columns)

# Iterate through all the rows in Dataframe
for row in df.itertuples(index=False):
    boredom_map[str(row.Boredom)] += 1
    engagement_map[str(row.Engagement)] += 1
    confusion_map[str(row.Confusion)] += 1
    frustration_map[str(row.Frustration)] += 1

# Display the counts
print("Boredom Map:", boredom_map)
print("Engagement Map:", engagement_map)
print("Confusion Map:", confusion_map)
print("Frustration Map:", frustration_map)

haarcascade = "haarcascades/haarcascade_frontalface_default.xml"

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(haarcascade)


# Specify the file you want to test
test_file = "C:/Users/ahmad/Desktop/EngagementML/DAiSEE/DataSet/Test/500067/5000671002/50006710024.jpg"  # Update with the actual file path
output_folder_cropped = "C:/Users/ahmad/Desktop/EngagementML/cropped_faces"
output_folder_gray = "C:/Users/ahmad/Desktop/EngagementML/cropped_gray_fraces"


def crop_face(image_path, output_dir_cropped, output_dir_gray):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {image_path}")
        return

    # Detect faces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for i, (x, y, w, h) in enumerate(faces):
        # Crop the face
        cropped_face = image[y:y + h, x:x + w]
        cropped_face = image[y:y + h, x:x + w]
        plt.imshow(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
        plt.title(f"Cropped Face {i}")
        plt.axis('off')
        plt.show()

        # Ensure output directories exist
        os.makedirs(output_dir_cropped, exist_ok=True)
        os.makedirs(output_dir_gray, exist_ok=True)

        # Save cropped face
        cropped_filename = os.path.join(output_dir_cropped, f"{os.path.basename(image_path)}_face{i}.jpg")
        cv2.imwrite(cropped_filename, cropped_face)

        # Convert to grayscale and save
        gray_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
        plt.imshow(gray_face, cmap='gray')  # Display grayscale image
        plt.title(f"Grayscale Face {i}")
        plt.axis('off')
        plt.show()

        gray_filename = os.path.join(output_dir_gray, f"{os.path.basename(image_path)}_gray_face{i}.jpg")
        cv2.imwrite(gray_filename, gray_face)

    if not faces.any():
        print(f"No faces detected in {image_path}")
