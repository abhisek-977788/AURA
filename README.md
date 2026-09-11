# AURA — Audio-Visual Unnatural Representation Analyzer
**Deepfake Detection System for Predictive Analysis (BCA Final Year Project)**

AURA is a multimodal deepfake detection platform designed for cybercrime investigations, digital forensics, and media authenticity verification. The platform analyzes human face images, facial video sequences, and synthetic voice recordings using three independent deep learning models, explainable artificial intelligence (Grad-CAM & Spectrogram Saliency), and a weighted multimodal decision fusion engine.

---

## 📁 Project Architecture

```text
AURA/
│
├── data/                               # Extracted dataset storage (unpacked automatically)
│   ├── image/                          # 140K Real vs Fake Human Faces (train/valid/test)
│   ├── video/                          # FaceForensics++ C23 (100 Real & 100 Fake balanced)
│   └── audio/                          # Voice Deepfakes & Audio conversions
│
├── preprocessing/                      # Feature extraction & dataset preparation
│   ├── image_preprocess.py             # 256x256 resizing, data augmentations, DataLoaders
│   ├── video_preprocess.py             # 10-frame extraction, MediaPipe face crop to 224x224
│   └── audio_preprocess.py             # 16kHz resampling, silence trim, Mel/LFCC/MFCC extraction
│
├── models/                             # Deep Learning Model Architectures
│   ├── image_model.py                  # Transfer learning with EfficientNet-B0 + Grad-CAM hooks
│   ├── video_model.py                  # Frame-level EfficientNet-B0 + temporal aggregation
│   └── audio_model.py                  # RawNet2 / Spectrogram-CNN 2D Residual Architecture
│
├── training/                           # Training Pipelines & Evaluation Utilities
│   ├── train_image.py                  # 20 Epochs, AdamW, BCEWithLogitsLoss, Early Stopping
│   ├── train_video.py                  # 15 Epochs, AdamW, BCEWithLogitsLoss, Per-Video Charts
│   ├── train_audio.py                  # 25 Epochs, AdamW, BCEWithLogitsLoss, Saliency Maps
│   └── utils.py                        # Metrics (AUC, F1, Acc), ROC & Loss Curves, Grad-CAM
│
├── saved_models/                       # Checkpoints of trained model weights (.pth)
│   ├── image_detector.pth
│   ├── video_detector.pth
│   └── audio_detector.pth
│
├── results/                            # Forensic evaluation outputs & visualizations
│   ├── confusion_matrix/               # Confusion matrix heatmaps
│   ├── roc_curves/                     # Receiver Operating Characteristic (ROC) curves
│   ├── gradcam/                        # Explainable AI Grad-CAM heatmaps & saliency maps
│   ├── training_logs/                  # Loss and accuracy progression graphs
│   └── metrics.csv                     # Historical evaluation metrics log
│
├── inference/                          # Independent Prediction & Multimodal Fusion
│   ├── predict_image.py                # Single image inference + Grad-CAM display
│   ├── predict_video.py                # Single video inference + frame confidence timeline
│   ├── predict_audio.py                # Single audio inference + Mel Spectrogram plot
│   └── multimodal_fusion.py            # Multimodal weighted fusion (0.30 Img, 0.40 Vid, 0.30 Aud)
│
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation & execution guide
```

---

## ⚙️ Installation & Environment Setup

### 1. Requirements
* Python 3.11 or higher
* NVIDIA GPU with CUDA support (e.g., RTX 4050 Laptop GPU) or CPU fallback
* Microsoft Windows 10/11

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Model Training Pipelines

Every training script runs completely independently, automatically extracts its respective dataset from the ZIP archives (`Image.zip`, `archive (8).zip`, `archive (6).zip`), evaluates on test/validation data, and saves the best model weights into `saved_models/`.

### 1. Image Model (Human Face Deepfake Detection)
* **Dataset**: 140K Real and Fake Faces (`Image.zip`)
* **Architecture**: EfficientNet-B0 (Transfer Learning)
* **Hyperparameters**: 20 Epochs, Batch Size 32, Learning Rate $10^{-4}$, AdamW, BCEWithLogitsLoss, CosineAnnealingLR.
```bash
python training/train_image.py --epochs 20 --batch-size 32
```
*Optional quick test with sample limit:*
```bash
python training/train_image.py --epochs 3 --batch-size 16 --sample-limit 500
```

### 2. Video Model (Deepfake Video Detection)
* **Dataset**: FaceForensics++ C23 (`archive (8).zip`)
* **Sample**: 100 Original Real videos and 100 Fake videos balanced across manipulation methods (`Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`, `FaceShifter`, `DeepFakeDetection`).
* **Processing**: 10 evenly spaced frames per video, MediaPipe face detection, cropped to $224 \times 224$.
* **Hyperparameters**: 15 Epochs, Batch Size 16, Learning Rate $10^{-4}$, AdamW, BCEWithLogitsLoss.
```bash
python training/train_video.py --epochs 15 --batch-size 16
```

### 3. Audio Model (Synthetic Voice & Cloned Speech Detection)
* **Dataset**: Voice Conversions & Cloned Audio (`archive (6).zip`)
* **Processing**: 16,000 Hz resampling, silence trimming, RMS normalization, Mel Spectrogram ($80$ filter bands, $1024$ FFT, $256$ hop).
* **Architecture**: RawNet2 / Spectrogram-CNN 2D Residual Network.
* **Hyperparameters**: 25 Epochs, Batch Size 32, Learning Rate $10^{-4}$, AdamW, BCEWithLogitsLoss.
```bash
python training/train_audio.py --epochs 25 --batch-size 32
```

---

## 🔍 Standalone Inference Scripts

### 1. Image Inference
Classifies any face image and generates an Explainable AI Grad-CAM heatmap highlighting manipulated pixels.
```bash
python inference/predict_image.py --input path/to/face_sample.jpg
```
*Outputs:*
* Prediction: `REAL` or `FAKE`
* Calibrated Confidence Percentage
* Heatmap saved to `results/gradcam/<image_name>_gradcam.png`

### 2. Video Inference
Evaluates an MP4 video file, detects faces across 10 temporal frames, generates a frame-by-frame confidence graph, and computes overall video authenticity.
```bash
python inference/predict_video.py --input path/to/video_sample.mp4
```
*Outputs:*
* Prediction: `REAL VIDEO` or `FAKE VIDEO`
* Overall Confidence Percentage
* Frame-by-frame confidence score bar chart
* Timeline graph saved to `results/training_logs/<video_name>_timeline.png`

### 3. Audio Inference
Evaluates a WAV/MP3 voice recording, extracts the Mel Spectrogram, and predicts synthetic speech.
```bash
python inference/predict_audio.py --input path/to/voice_sample.wav
```
*Outputs:*
* Prediction: `REAL AUDIO` or `FAKE AUDIO`
* Confidence Percentage
* Spectrogram visualization saved to `results/training_logs/<audio_name>_spectrogram.png`

---

## 🔗 Multimodal Decision Fusion

The multimodal fusion module combines confidence scores from all three modalities using weighted average fusion:
$$\text{Score}_{\text{final}} = (0.30 \times P_{\text{image}}) + (0.40 \times P_{\text{video}}) + (0.30 \times P_{\text{audio}})$$

### CLI Execution:
```bash
python inference/multimodal_fusion.py --image-prob 0.95 --video-prob 0.98 --audio-prob 0.05
```

### Output JSON Format:
```json
{
  "image_prediction": "Fake",
  "video_prediction": "Fake",
  "audio_prediction": "Real",
  "final_prediction": "Fake",
  "confidence": 69.2
}
```

You can also pass media files directly to run end-to-end multimodal inference:
```bash
python inference/multimodal_fusion.py --image-file sample.jpg --video-file sample.mp4 --audio-file sample.wav
```

---

## 🔬 Explainable AI (XAI) & Deliverables

1. **Grad-CAM (Gradient-weighted Class Activation Mapping)**: Overlays visual heatmaps on fake faces and video frames to reveal which facial features (eyes, blending seams, mouth boundary) triggered the model.
2. **Spectrogram Saliency Attribution**: Visualizes anomalous high-frequency vocoder artifacts in synthetic speech.
3. **Forensic Artifacts Generated**:
   * `results/metrics.csv`: Tabulated Accuracy, Precision, Recall, F1, ROC-AUC.
   * `results/roc_curves/`: True Positive vs False Positive trade-off curves.
   * `results/confusion_matrix/`: Type I and Type II error analysis heatmaps.
   * `results/training_logs/`: Epoch-by-epoch loss convergence curves.

---

## 🎓 Academic Presentation Notes (BCA Project Defense)

* **Why EfficientNet-B0?** Compound scaling balances depth, width, and resolution with high inference speed ($<15$ ms per frame) and low memory footprint on modern GPUs.
* **Why RawNet2 / Spectrogram-CNN?** AI voice cloning (ElevenLabs, Tortoise, VALL-E) leaves subtle phase and spectral discontinuities in log-mel spectrograms that residual convolutional filters reliably capture.
* **Multimodal Robustness**: Fusing image ($30\%$), video ($40\%$), and audio ($30\%$) ensures that a sophisticated deepfake altering only one modality (e.g., audio dubbing or face-swapping alone) is accurately identified.
