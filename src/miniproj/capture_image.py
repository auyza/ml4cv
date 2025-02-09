import cv2
import mediapipe as mp

def detect_and_capture_face(output_filename="captured_face.jpg", face_detection_confidence=0.5):
    """Detects a face in the webcam feed and captures a frame if a face is detected.

    Args:
        output_filename (str): The name of the file to save the captured image to.
        face_detection_confidence (float): The minimum confidence level for face detection.
    """

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=face_detection_confidence
    )

    cap = cv2.VideoCapture(0)  # 0 usually refers to the default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame to RGB (MediaPipe needs RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(rgb_frame)

        if results.detections:  # If a face is detected
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                h, w, _ = frame.shape  # Get image dimensions
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Clip the bounding box to image boundaries (important!)
                x = max(0, x)
                y = max(0, y)
                width = min(w - x, width)
                height = min(h - y, height)

                if width > 0 and height > 0: # Check if the bounding box is valid
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)  # Draw rectangle
                    cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    key = cv2.waitKey(1) # Wait for a key press
                    if key == ord('c'): # Press 'c' to capture
                        cropped_face = frame[y:y+height, x:x+width]
                        cv2.imwrite(output_filename, cropped_face)
                        print(f"Face captured and saved to {output_filename}")
                        break # Exit the inner loop after capturing
            if key == ord('c'): # Exit the outer loop after capturing
                break

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    face_detection.close()

# Example usage:
detect_and_capture_face("my_face_capture.jpg", 0.7)  # Save as my_face_capture.jpg, confidence 70%