import tensorflow as tf
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
import os
import glob  # Import the glob module!  This is the fix.

# 1. Data Collection and Preparation (Simplified Example - Replace with your actual data loading)
# This example creates dummy data.  In a real application, you would load your data from files.
def load_data(data_dir, label_map=None): # Added label_map
    images = []
    labels = []

    subdirs = glob.glob(os.path.join(data_dir, "*"))

    for subdir in subdirs:
        if os.path.isdir(subdir):
            label_name = os.path.basename(subdir)

            if label_map is None: # If no label_map provided, use folder name as label
                label = label_name
            else: # Use label_map to convert folder name to label
                if label_name in label_map:
                    label = label_map[label_name]
                else:
                    print(f"Skipping directory {subdir}: Label not found in label_map.")
                    continue

            for image_file in glob.glob(os.path.join(subdir, "*")):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    img = cv2.imread(image_file)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                        labels.append(label)
    return images, labels


# 2. Face Detection (MediaPipe)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def detect_faces(image):
    results = face_detection.process(image)
    faces = []
    if results.detections:
        h, w, _ = image.shape  # Get image dimensions ONCE

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Clip the bounding box to image boundaries (THE FIX):
            x = max(0, x)  # Ensure x is not less than 0
            y = max(0, y)  # Ensure y is not less than 0
            width = min(w - x, width)  # Ensure width doesn't go beyond image right edge
            height = min(h - y, height)  # Ensure height doesn't go beyond image bottom edge

            # Check for valid bounding box AFTER clipping:
            if width > 0 and height > 0:  # Only append if width and height are still positive
                faces.append((x, y, width, height))
            else:
                print(f"Invalid bounding box after clipping: {x}, {y}, {width}, {height}. Skipping.")

    return faces

# 3. Data Preprocessing (including face cropping and resizing)
def preprocess_face(image, bbox, target_size=(128, 128)):
  x, y, w, h = bbox
  face_img = image[y:y+h, x:x+w]
  face_img_resized = cv2.resize(face_img, target_size)
  face_img_normalized = face_img_resized / 255.0 # Normalize pixel values
  return face_img_normalized

# 4. Model Training (using MobileNetV3Small as an example)
def train_smile_model(x_train, y_train, x_val, y_val, epochs=10, batch_size=32):
    base_model = tf.keras.applications.MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False  # Freeze the base model initially

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output: probability of smiling
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) # Add early stopping

    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])

    return model

# 5. Main Execution
if __name__ == "__main__":
    data_dir = "data/datasets/mini-proj/original/" # Replace with the path to your data directory
    # Define a label map to convert folder names to labels:
    label_map = {
        "non_smile": 0,  # Map "non_smile" folder to label 0
        "smile": 1,      # Map "smile" folder to label 1
        "test": 2       # Map "test" folder to label 2 (or whatever you want)
    }

    print("[INFO] start load data .. ")
    images, labels = load_data(data_dir, label_map) # Pass the label_map
    # images, labels = load_data(data_dir)
    
    print("[INFO] completed load data .. ")
    print("[INFO] start detect face image .. ")

    face_images = []
    face_labels = []

    for image, label in zip(images, labels):
        faces = detect_faces(image)
        if faces: # If faces are detected
            for face in faces:
                processed_face = preprocess_face(image, face)
                face_images.append(processed_face)
                face_labels.append(label) # Use original label for the face

    print("[INFO] start split data .. ")
    x_train, x_val, y_train, y_val = train_test_split(np.array(face_images), np.array(face_labels), test_size=0.2, random_state=42)
    
    print("[INFO] start start train model .. ")
    smile_model = train_smile_model(x_train, y_train, x_val, y_val)
    smile_model.save("src/miniproj/model_save/smile_detection_model.keras") # Save the trained model
    
    print("[INFO] Model trained and saved!")

    # Example of loading and using the saved model:
    # loaded_model = tf.keras.models.load_model("smile_detection_model")
    # ... (Use loaded_model for inference) ...