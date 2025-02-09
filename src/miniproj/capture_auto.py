import cv2
import mediapipe as mp
import time
import os

def detect_and_capture_face(output_dir="captured_faces", face_detection_confidence=0.5, capture_interval=60):  # capture_interval in seconds
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

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    last_capture_time = 0  # Initialize last capture time

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
                    cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    current_time = time.time()
                    if current_time - last_capture_time >= capture_interval:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")  # Get timestamp
                        output_filename = os.path.join(output_dir, f"captured_face_{timestamp}.jpg")
                        cropped_face = frame[y:y+height, x:x+width]
                        cv2.imwrite(output_filename, cropped_face)
                        print(f"Face captured and saved to {output_filename}")
                        last_capture_time = current_time  # Update last capture time

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_detection.close()

# Example usage:
detect_and_capture_face(capture_interval=10)  # Capture every 60 seconds (1 minute)