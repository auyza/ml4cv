import cv2
import mediapipe as mp
import time
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model('src/miniproj/model_save/trained_model.hdf5') 

# 3. Classify a Frame
def classify_frame(frame, model, class_names):
    img_array = preprocess_frame(frame)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    return predicted_class_name, confidence

def get_class_names(train_dir="data/datasets/mini-proj/process"):
  class_names = os.listdir(train_dir)
  valid_classes = []
  for class_name in class_names:
    if not class_name.startswith('.'):
      class_path = os.path.join(train_dir, class_name)
      if os.path.isdir(class_path):
        valid_classes.append(class_name)
  return valid_classes

def detect_face_smile(input_face, class_names, target_size=(224, 224)):
    frame = cv2.resize(input_face, target_size)
    img_array = image.img_to_array(frame)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch dimension
    img_array /= 255.0  # Normalize pixel values
    
    # class_names = get_class_names()
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    # print(f"Image: {img_path}")
    print(f"Predicted Class: {predicted_class_name}")
    print(f"Confidence: {confidence:.4f}")
    
    return f"{predicted_class_name} : {confidence:.4f}"



def detect_and_capture_face(output_dir="data/datasets/output/captured_faces", face_detection_confidence=0.5, capture_interval=60):  # capture_interval in seconds
    """Detects faces and auto-captures at specified intervals if a face is detected.

    Args:
        output_dir (str): Directory to save captured images.
        face_detection_confidence (float): Minimum confidence for face detection.
        capture_interval (int): Time interval (in seconds) between captures.
    """

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=face_detection_confidence
    )
    
    class_names = get_class_names()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    last_capture_time = 0  # Initialize last capture time
    last_detact_result = "Just smail will make you feel bester !!!"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                x = max(0, x)
                y = max(0, y)
                width = min(w - x, width)
                height = min(h - y, height)

                if width > 0 and height > 0:
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(frame, last_detact_result, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    # cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    current_time = time.time()
                    if current_time - last_capture_time >= capture_interval:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")  # Get timestamp
                        output_filename = os.path.join(output_dir, f"face_{timestamp}.jpg")
                        cropped_face = frame[y:y+height, x:x+width]
                        last_detact_result = detect_face_smile(cropped_face , class_names)
                        cv2.putText(cropped_face, last_detact_result,(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.imwrite(output_filename, cropped_face)
                        # print(f"Face captured and saved to {output_filename}")
                        last_capture_time = current_time  # Update last capture time

        # cv2.imshow("Face Detection", frame)
        cv2.imshow(last_detact_result, frame)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_detection.close()

# Example usage:
detect_and_capture_face(capture_interval=5)  # Capture every 60 seconds (1 minute)