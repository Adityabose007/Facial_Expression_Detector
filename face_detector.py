import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Configure the Face Detector using the modern Tasks API
model_path = 'blaze_face_short_range.tflite'

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

# Initialize Webcam
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not video_capture.isOpened():
    video_capture = cv2.VideoCapture(0)

# Setup directories for capturing crops
output_crop_dir = "captured_faces"
os.makedirs(output_crop_dir, exist_ok=True)

# Variables for FPS calculations and feature toggles
prev_frame_time = 0
blur_background = False
neon_mode = True

print("\n" + "=" * 50)
print("[CONTROLS] Interactive Hotkeys on Video Window:")
print("  Press 'b' : Toggle Background Blur")
print("  Press 'n' : Toggle Neon/Sci-Fi Box Effects")
print("  Press 'c' : Save/Crop detected faces to disk")
print("  Press 'ESC': Exit Application")
print("=" * 50 + "\n")

# Initialize the detector context manager
with vision.FaceDetector.create_from_options(options) as detector:
    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            time.sleep(0.1)
            continue

        # Flip horizontally for natural selfie view
        frame = cv2.flip(frame, 1)
        img_height, img_width, _ = frame.shape

        # Keep a clean, unaltered copy of the frame for cropping out faces cleanly
        raw_backup_frame = frame.copy()

        # Feature: Background Blur (Creates a bokeh/portrait effect outside the face bounding box)
        if blur_background:
            blurred_bg = cv2.GaussianBlur(frame, (41, 41), 0)
            # Create a black mask, we'll cut holes where the faces are
            face_mask = cv2.zeros_like(frame)
        else:
            face_mask = None

        # Convert BGR (OpenCV) to RGB (MediaPipe requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms = int(time.time() * 1000)

        # Run inference
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
        face_count = 0

        # Process results if faces are found
        if detection_result.detections:
            face_count = len(detection_result.detections)

            for index, detection in enumerate(detection_result.detections):
                bbox = detection.bounding_box

                # Fast boundaries clipping to avoid out-of-bounds array crashes
                x = max(0, bbox.origin_x)
                y = max(0, bbox.origin_y)
                w = min(bbox.width, img_width - x)
                h = min(bbox.height, img_height - y)

                # Build mask for background blur feature if active
                if blur_background:
                    cv2.rectangle(face_mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

                # Draw Visual Elements
                if neon_mode:
                    # Attractive Sci-Fi Corner Bracket styling
                    line_color = (0, 255, 255)  # Bright Neon Yellow/Cyan
                    thickness = 2
                    length = int(w * 0.2)  # Corner piece size proportional to face

                    # Top Left
                    cv2.line(frame, (x, y), (x + length, y), line_color, thickness + 2)
                    cv2.line(frame, (x, y), (x, y + length), line_color, thickness + 2)
                    # Top Right
                    cv2.line(frame, (x + w, y), (x + w - length, y), line_color, thickness + 2)
                    cv2.line(frame, (x + w, y), (x + w, y + length), line_color, thickness + 2)
                    # Bottom Left
                    cv2.line(frame, (x, y + h), (x + length, y + h), line_color, thickness + 2)
                    cv2.line(frame, (x, y + h), (x, y + h - length), line_color, thickness + 2)
                    # Bottom Right
                    cv2.line(frame, (x + w, y + h), (x + w - length, y + h), line_color, thickness + 2)
                    cv2.line(frame, (x + w, y + h), (x + w, y + h - length), line_color, thickness + 2)

                    # Transparent tint fill overlay inside the face box
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), -1)
                    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
                else:
                    # Classic UI
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Overlay customized text tags for individual target tracks
                cv2.putText(frame, f"TARGET #{index + 1}", (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if neon_mode else (255, 0, 0), 2)

                # Extract and draw keypoint landmarks
                if detection.keypoints:
                    for kp_idx, keypoint in enumerate(detection.keypoints):
                        kx = int(keypoint.x * img_width)
                        ky = int(keypoint.y * img_height)
                        # Alternate landmark color design (Glow green cores)
                        cv2.circle(frame, (kx, ky), 4, (0, 255, 0), -1)
                        cv2.circle(frame, (kx, ky), 5, (255, 255, 255), 1)

        # Merge portrait background blur if enabled
        if blur_background and detection_result.detections:
            # Combine original targets frame with blurred environment using the mask
            frame = np.where(face_mask == 255, frame, blurred_bg)
        elif blur_background and not detection_result.detections:
            frame = blurred_bg

        # Calculate exact System Runtime FPS Performance
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time

        # Head-Up Display (HUD) Styling Banner
        cv2.rectangle(frame, (0, 0), (240, 75), (20, 20, 20), -1)  # Dark transparent header backing box
        cv2.putText(frame, f"FPS  : {int(fps)}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"FACES: {face_count}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)

        # Display output
        cv2.imshow('Modern MediaPipe Face Detection', frame)

        # Keyboard Interactivity Listener
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC Key to quit
            break
        elif key == ord('b'):  # Toggle background blur
            import numpy as np  # Handled gracefully safely during dynamic toggling

            blur_background = not blur_background
            print(f"[TOGGLE] Background portrait blur set to: {blur_background}")
        elif key == ord('n'):  # Toggle Cyberpunk Neon visual HUD UI
            neon_mode = not neon_mode
        elif key == ord('c'):  # Crop and save current faces out to high-res disk files
            if detection_result.detections:
                for idx, d in enumerate(detection_result.detections):
                    b = d.bounding_box
                    cx = max(0, b.origin_x)
                    cy = max(0, b.origin_y)
                    cw = min(b.width, img_width - cx)
                    ch = min(b.height, img_height - cy)

                    cropped_face = raw_backup_frame[cy:cy + ch, cx:cx + cw]
                    if cropped_face.size > 0:
                        filename = f"{output_crop_dir}/face_{int(time.time())}_{idx}.jpg"
                        cv2.imwrite(filename, cropped_face)
                print(
                    f"[CAPTURE] Extracted and saved {len(detection_result.detections)} face(s) safely to folder: /{output_crop_dir}")
            else:
                print("[CAPTURE] No faces present in stream view to extract right now.")

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
print("[INFO] Application closed cleanly.")