# CyberMesh: Real-Time Facial Expression & Emotion Detector

An interactive, high-performance computer vision application that detects human faces and classifies facial expressions in real time. Powered by **MediaPipe’s modern Face Landmarker API** and **OpenCV**, this project bypasses bulky deep learning frameworks by utilizing precise 3D blendshape coefficients to infer expressions with zero tracking latency.

Featuring a modular rule engine based on psychological expression matrices, the application tracks up to 8 distinct universal emotional states while providing a sleek, interactive cyber-themed HUD layer.

---

## 🚀 Features

* **Real-Time Expression Matrix**: Classifies 8 distinct facial configurations:
    * `NEUTRAL`, `HAPPY`, `SAD`, `SURPRISE`, `FEAR`, `ANGER`, `DISGUST`, and `CONTEMPT`.
* **Aesthetic Cyberpunk HUD**: Dynamically adapts interface border colors and transparent overlay hues based on the active emotion detected.
* **Background Portrait Blur**: Real-time bokeh/portrait mode masking that separates targets from their environments smoothly using custom alpha blending.
* **High-Res Target Cropping**: On-demand disk storage that crops, isolates, and exports clean target snapshots mapped to active emotion filenames.
* **Multi-Face Tracking**: Concurrently isolates and monitors landmarks for up to 4 targets simultaneously without dipping frame execution speeds.

---

## 🛠️ Tech Stack & Architecture

* **Core Language:** Python 3.10+
* **Inference Engine:** MediaPipe Tasks API (Face Landmarker Model Bundle)
* **Graphics Pipeline:** OpenCV (Open Source Computer Vision Library)
* **Data Layout Processing:** NumPy

The application reads facial geometries through **52 distinct blendshape parameters** (such as `mouthSmileRight`, `jawOpen`, and `browDownLeft`). These parameters represent individual muscle movements ranging from `0.0` (at rest) to `1.0` (fully flexed), creating an instant, lightweight decision matrix.

---

## 📥 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/yourusername/facial-expression-detector.git](https://github.com/yourusername/facial-expression-detector.git)
   cd facial-expression-detector

   Install Core Dependencies-----
   
   pip install opencv-python mediapipe numpy

   # Using wget terminal utility
wget -O face_landmarker.task [https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
   
   🎮 How To Run & Interactive HotkeysExecute the master script structure to run the processing window:Bashpython comprehensive_expression_detector.py
   
Once the window opens, tap any of the interactive operational hotkeys on your keyboard to switch features on the fly:KeyAction RoutineDescriptionESCExit ApplicationCleanly releases hardware video buffers and closes graphic frames.bToggle Background BlurSpatially masks out background environments using a Gaussian pass.nToggle Neon FX ModeToggles between minimal boxes and glowing reactive sci-fi frames.cCapture ExpressionsCuts face frames from the clean raw array and outputs tagged .jpg files.📂 Project Structure OverviewPlaintext├── comprehensive_expression_detector.py   # Main application source code
├── face_landmarker.task                    # MediaPipe model weight asset bundle
├── captured_expressions/                  # Auto-generated crop storage folder
│   ├── face_HAPPY_171492312_0.jpg         # Example exported capture file
│   └── face_SURPRISE_171492355_1.jpg
└── README.md                              # Repository Documentation

📝 LicenseDistributed under the MIT License. See LICENSE for more information.
