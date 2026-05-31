import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Configure the Face Landmarker using the modern Tasks API
# Make sure 'face_landmarker.task' is downloaded in your script directory
model_path = 'face_landmarker.task'

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,  # Crucial for extracting muscle movement variables
    output_facial_transformation_matrixes=False,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=4  # Supports tracking multiple targets simultaneously
)

# Initialize Webcam
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not video_capture.isOpened():
    video_capture = cv2.VideoCapture(0)

# Setup directories for capturing crops
output_crop_dir = "captured_expressions"
os.makedirs(output_crop_dir, exist_ok=True)

# Variables for FPS calculations and feature toggles
prev_frame_time = 0
blur_background = False
neon_mode = True

print("\n" + "=" * 50)
print("[CONTROLS] Interactive Hotkeys on Video Window:")
print("  Press 'b' : Toggle Background Blur")
print("  Press 'n' : Toggle Neon/Sci-Fi Box Effects")
print("  Press 'c' : Save/Crop detected expressions to disk")
print("  Press 'ESC': Exit Application")
print("=" * 50 + "\n")


def estimate_all_emotions(blendshapes):
    """
    Analyzes MediaPipe Blendshapes to distinguish between all primary universal facial expressions.
    Returns: (Emotion Name String, BGR Color Tuple for visual effects)
    """
    # Create a lookup dictionary mapping muscle shape names to their 0.0 - 1.0 activation values
    scores = {b.category_name: b.score for b in blendshapes}

    # --- Extract Muscle Group Values ---
    # Smiling (Zygomaticus major)
    smile_r = scores.get("mouthSmileRight", 0.0)
    smile_l = scores.get("mouthSmileLeft", 0.0)
    avg_smile = (smile_l + smile_r) / 2.0

    # Frowning (Depressor anguli oris)
    frown_r = scores.get("mouthFrownRight", 0.0)
    frown_l = scores.get("mouthFrownLeft", 0.0)
    avg_frown = (frown_l + frown_r) / 2.0

    # Jaw Open & Mouth Stretch
    jaw_open = scores.get("jawOpen", 0.0)
    mouth_funnel = scores.get("mouthFunnel", 0.0)
    mouth_pucker = scores.get("mouthPucker", 0.0)
    mouth_stretch_r = scores.get("mouthStretchRight", 0.0)
    mouth_stretch_l = scores.get("mouthStretchLeft", 0.0)
    avg_mouth_stretch = (mouth_stretch_r + mouth_stretch_l) / 2.0

    # Eyebrows Down / Furrowed (Corrugator supercilii)
    brow_down_r = scores.get("browDownRight", 0.0)
    brow_down_l = scores.get("browDownLeft", 0.0)
    avg_brow_down = (brow_down_r + brow_down_l) / 2.0

    # Eyebrows Raised (Frontalis)
    brow_inner_up = scores.get("browInnerUp", 0.0)
    brow_outer_up_r = scores.get("browOuterUpRight", 0.0)
    brow_outer_up_l = scores.get("browOuterUpLeft", 0.0)
    avg_brow_outer_up = (brow_outer_up_r + brow_outer_up_l) / 2.0

    # Eyes Wide (Levator palpebrae superioris)
    eye_wide_r = scores.get("eyeWideRight", 0.0)
    eye_wide_l = scores.get("eyeWideLeft", 0.0)
    avg_eye_wide = (eye_wide_r + eye_wide_l) / 2.0

    # Nose Wrinkling & Upper Lip Raiser (Levator labii superioris alaeque nasi - Disgust)
    nose_sneer_r = scores.get("noseSneerRight", 0.0)
    nose_sneer_l = scores.get("noseSneerLeft", 0.0)
    avg_nose_sneer = (nose_sneer_r + nose_sneer_l) / 2.0

    # Lip Corner Pullers (Asymmetry indicators for Contempt)
    dimple_r = scores.get("mouthDimpleRight", 0.0)
    dimple_l = scores.get("mouthDimpleLeft", 0.0)

    # --- Comprehensive Decision Rule-Engine Matrix ---

    # 1. SURPRISE: High wide eyes, lifted brows, dropped open jaw
    if avg_eye_wide > 0.45 and (brow_inner_up > 0.4 or avg_brow_outer_up > 0.4) and jaw_open > 0.25:
        return "SURPRISE", (0, 255, 255)  # Bright Yellow/Cyan

    # 2. FEAR: Wide eyes, tense stretched mouth pulling outward, brows up/together
    elif avg_eye_wide > 0.35 and avg_mouth_stretch > 0.4 and (brow_inner_up > 0.3 or avg_brow_down > 0.2):
        return "FEAR", (255, 0, 128)  # Deep Pink / Violet

    # 3. DISGUST: Nose sneer wrinkling, upper lip raised, often combined with a bit of a frown
    elif avg_nose_sneer > 0.45 or (scores.get("mouthUpperUpRight", 0.0) > 0.5 and avg_nose_sneer > 0.2):
        return "DISGUST", (0, 128, 0)  # Olive / Dark Green

    # 4. ANGER: Heavy brow furrow lowering, combined with tight lips or open yelling jaw
    elif avg_brow_down > 0.45 and (avg_frown > 0.25 or mouth_pucker > 0.3 or jaw_open > 0.3) and avg_smile < 0.1:
        return "ANGER", (0, 0, 255)  # Pure Alert Red

    # 5. HAPPY: Broad symmetrical lip corner pullers active
    elif avg_smile > 0.40:
        return "HAPPY", (0, 255, 0)  # Emerald Green

    # 6. SAD: Downward lip corners, inner eyebrows pulled upwards towards the center
    elif avg_frown > 0.40 or (brow_inner_up > 0.45 and avg_smile < 0.15) or (scores.get("mouthRollLower", 0.0) > 0.4):
        return "SAD", (255, 0, 0)  # Ocean Blue

    # 7. CONTEMPT: Marked asymmetric smile/dimple elevation on only one side of the mouth
    elif abs(dimple_r - dimple_l) > 0.45 or abs(smile_r - smile_l) > 0.5:
        return "CONTEMPT", (0, 140, 255)  # Orange / Amber

    # 8. SHOCKED/FUNNEL: Exaggerated O-shaped mouth pucker
    elif mouth_funnel > 0.5 and jaw_open > 0.2:
        return "SHOCKED", (255, 0, 255)  # Magenta

    # DEFAULT. NEUTRAL: Face muscles resting
    else:
        return "NEUTRAL", (180, 180, 180)  # Light Gray


# Initialize the face landmarker context manager
with vision.FaceLandmarker.create_from_options(options) as landmarker:
    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            time.sleep(0.1)
            continue

        # Flip frame horizontally for mirror reflection appearance
        frame = cv2.flip(frame, 1)
        img_height, img_width, _ = frame.shape

        # Clean frame copy backup for artifact-free crops
        raw_backup_frame = frame.copy()

        # Feature: Background Blur processing mask setup
        if blur_background:
            blurred_bg = cv2.GaussianBlur(frame, (41, 41), 0)
            face_mask = np.zeros_like(frame)
        else:
            face_mask = None

        # Convert BGR format to RGB framework
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms = int(time.time() * 1000)

        # Run Face Mesh and Blendshape Inference
        landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        face_count = 0

        # Process when face landmarks are present
        if landmarker_result.face_landmarks:
            face_count = len(landmarker_result.face_landmarks)

            for index, landmarks in enumerate(landmarker_result.face_landmarks):
                # Enclose the extreme facial edge landmarks to calculate bounding boxes
                x_coordinates = [lm.x * img_width for lm in landmarks]
                y_coordinates = [lm.y * img_height for lm in landmarks]

                x_min, x_max = int(min(x_coordinates)), int(max(x_coordinates))
                y_min, y_max = int(min(y_coordinates)), int(max(y_coordinates))

                # Apply dynamic bounding margins to framed head structure
                padding_x = int((x_max - x_min) * 0.12)
                padding_y = int((y_max - y_min) * 0.12)

                x = max(0, x_min - padding_x)
                y = max(0, y_min - padding_y)
                w = min(x_max + padding_x, img_width) - x
                h = min(y_max + padding_y, img_height) - y

                # Fallback defaults
                current_emotion = "NEUTRAL"
                theme_color = (180, 180, 180)

                # Extract and parse individual face expression matrices
                if landmarker_result.face_blendshapes and len(landmarker_result.face_blendshapes) > index:
                    blendshapes = landmarker_result.face_blendshapes[index]
                    current_emotion, theme_color = estimate_all_emotions(blendshapes)

                # Segment face space to stay clear of background blur
                if blur_background:
                    cv2.rectangle(face_mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

                # Draw Interface Visual Design Elements
                if neon_mode:
                    # Cyberpunk themed outer corner locks calibrated to look matching with expression states
                    thickness = 2
                    length = int(w * 0.2)

                    # Top Left corner brackets
                    cv2.line(frame, (x, y), (x + length, y), theme_color, thickness + 2)
                    cv2.line(frame, (x, y), (x, y + length), theme_color, thickness + 2)
                    # Top Right corner brackets
                    cv2.line(frame, (x + w, y), (x + w - length, y), theme_color, thickness + 2)
                    cv2.line(frame, (x + w, y), (x + w, y + length), theme_color, thickness + 2)
                    # Bottom Left corner brackets
                    cv2.line(frame, (x, y + h), (x + length, y + h), theme_color, thickness + 2)
                    cv2.line(frame, (x, y + h), (x, y + h - length), theme_color, thickness + 2)
                    # Bottom Right corner brackets
                    cv2.line(frame, (x + w, y + h), (x + w - length, y + h), theme_color, thickness + 2)
                    cv2.line(frame, (x + w, y + h), (x + w, y + h - length), theme_color, thickness + 2)

                    # Light internal tint overlay
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), theme_color, -1)
                    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
                else:
                    # Standard Minimal Box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), theme_color, 2)

                # Render dynamic status string above the head tracking area
                text_label = f"ID:{index + 1} | {current_emotion}"
                cv2.putText(frame, text_label, (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, theme_color, 2)

                # Draw clean, stylized landmark mesh points (sampled to preserve high FPS rendering speed)
                for idx, lm in enumerate(landmarks):
                    if idx % 15 == 0:  # Space out points to look sleek and non-intrusive
                        kx = int(lm.x * img_width)
                        ky = int(lm.y * img_height)
                        cv2.circle(frame, (kx, ky), 2, theme_color, -1)

        # Merge background portrait blur layers if state is active
        if blur_background and landmarker_result.face_landmarks:
            frame = np.where(face_mask == 255, frame, blurred_bg)
        elif blur_background and not landmarker_result.face_landmarks:
            frame = blurred_bg

        # Frame Processing Speed Metrics Calculation (FPS)
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time

        # Draw dark aesthetic Status HUD Header overlay
        cv2.rectangle(frame, (0, 0), (240, 75), (20, 20, 20), -1)
        cv2.putText(frame, f"FPS  : {int(fps)}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"FACES: {face_count}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)

        # Show Output stream
        cv2.imshow('Modern MediaPipe Expression Detection', frame)

        # Interactivity Control Loops
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC Key to quit safely
            break
        elif key == ord('b'):
            blur_background = not blur_background
            print(f"[TOGGLE] Background portrait blur set to: {blur_background}")
        elif key == ord('n'):
            neon_mode = not neon_mode
        elif key == ord('c'):  # Crop out expression to disk folder
            if landmarker_result.face_landmarks:
                for idx, landmarks in enumerate(landmarker_result.face_landmarks):
                    x_coordinates = [lm.x * img_width for lm in landmarks]
                    y_coordinates = [lm.y * img_height for lm in landmarks]
                    x_min, x_max = int(min(x_coordinates)), int(max(x_coordinates))
                    y_min, y_max = int(min(y_coordinates)), int(max(y_coordinates))

                    cx = max(0, x_min - 20)
                    cy = max(0, y_min - 20)
                    cw = min(x_max + 20, img_width) - cx
                    ch = min(y_max + 20, img_height) - cy

                    cropped_face = raw_backup_frame[cy:cy + ch, cx:cx + cw]
                    if cropped_face.size > 0:
                        if landmarker_result.face_blendshapes and len(landmarker_result.face_blendshapes) > idx:
                            emo_string, _ = estimate_all_emotions(landmarker_result.face_blendshapes[idx])
                        else:
                            emo_string = "UNKNOWN"

                        filename = f"{output_crop_dir}/face_{emo_string}_{int(time.time())}_{idx}.jpg"
                        cv2.imwrite(filename, cropped_face)
                print(f"[CAPTURE] Successfully exported expressions to folder: /{output_crop_dir}")
            else:
                print("[CAPTURE] No target tracked inside the camera frame right now.")

# Window termination sequences
video_capture.release()
cv2.destroyAllWindows()
print("[INFO] Engine terminated cleanly.")