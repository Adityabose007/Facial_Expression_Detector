import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Configure the Face Detector using the modern Tasks API
# Make sure 'blaze_face_short_range.tflite' is in your script's folder
model_path = 'blaze_face_short_range.tflite'

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO  # Optimized specifically for real-time video streams
)

# Initialize Webcam with standard Windows DirectShow fallback to speed up hardware connection
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Fallback to standard initialization if DirectShow fails
if not video_capture.isOpened():
    video_capture = cv2.VideoCapture(0)

# Initialize the detector context manager
with vision.FaceDetector.create_from_options(options) as detector:
    print("\n[INFO] Camera pipeline active. Press 'ESC' on the video window to exit.")

    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            print("Ignoring empty camera frame. Waiting for device...")
            time.sleep(0.1)
            continue

        # Flip horizontally for natural selfie view
        frame = cv2.flip(frame, 1)

        # Convert BGR (OpenCV) to RGB (MediaPipe requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the NumPy frame into a MediaPipe Image Object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # The Video running mode requires a monotonically increasing timestamp in milliseconds
        frame_timestamp_ms = int(time.time() * 1000)

        # Run inference
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        # Draw results if a face is found
        if detection_result.detections:
            img_height, img_width, _ = frame.shape

            for detection in detection_result.detections:
                # Get bounding box coordinates
                bbox = detection.bounding_box
                x = bbox.origin_x
                y = bbox.origin_y
                w = bbox.width
                h = bbox.height

                # Draw the bounding box rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

                # Extract and draw keypoint landmarks (eyes, nose, mouth, ears)
                if detection.keypoints:
                    for keypoint in detection.keypoints:
                        # Keypoints are normalized (0.0 to 1.0), convert back to actual pixel locations
                        kx = int(keypoint.x * img_width)
                        ky = int(keypoint.y * img_height)
                        cv2.circle(frame, (kx, ky), 4, (0, 255, 0), -1)

        # Display output
        cv2.imshow('Modern MediaPipe Face Detection', frame)

        # Listen for the 'ESC' key to break the loop
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
print("[INFO] Application closed cleanly.")