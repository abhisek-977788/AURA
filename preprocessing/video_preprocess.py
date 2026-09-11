"""
AURA - Video Preprocessing Pipeline
Extracts 100 Real and 100 Fake videos from FaceForensics++ (archive (8).zip),
extracts 10 evenly-spaced frames per video, performs MediaPipe face detection & cropping,
and produces standardized 224x224 RGB face tensors.
"""

import os
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

try:
    import mediapipe as mp
    MP_FACE_AVAILABLE = True
except (ImportError, OSError, Exception):
    mp = None
    MP_FACE_AVAILABLE = False

VIDEO_ZIP_PATH = Path("D:/AURA/archive (8).zip")
VIDEO_EXTRACT_DIR = Path("D:/AURA/data/video")
CROPS_CACHE_DIR = Path("D:/AURA/data/video/face_crops")

MANIPULATION_FOLDERS = [
    "DeepFakeDetection",
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
]


def extract_balanced_videos(
    zip_path: Path = VIDEO_ZIP_PATH,
    dest_dir: Path = VIDEO_EXTRACT_DIR,
    real_count: int = 100,
    fake_count: int = 100,
) -> Dict[str, List[Path]]:
    """
    Selectively extracts 100 original videos and 100 fake videos (balanced across
    manipulation categories) from FaceForensics++ archive (8).zip into dest_dir.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    real_dir = dest_dir / "real"
    fake_dir = dest_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    existing_real = list(real_dir.glob("*.mp4"))
    existing_fake = list(fake_dir.glob("**/*.mp4"))

    if len(existing_real) >= real_count and len(existing_fake) >= fake_count:
        print(f"[Video Preprocess] Already extracted {len(existing_real)} real and {len(existing_fake)} fake videos.")
        return {"real": existing_real[:real_count], "fake": existing_fake[:fake_count]}

    if not zip_path.exists():
        raise FileNotFoundError(f"Video archive not found at: {zip_path}")

    print(f"[Video Preprocess] Scanning {zip_path.name} for 100 Real and 100 Fake videos...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        all_names = archive.namelist()

        # 1. Real videos from 'original'
        orig_names = [n for n in all_names if "/original/" in n.lower() and n.endswith(".mp4")]
        selected_real = orig_names[:real_count]

        # 2. Fake videos distributed across manipulation folders
        per_folder_count = (fake_count + len(MANIPULATION_FOLDERS) - 1) // len(MANIPULATION_FOLDERS)
        selected_fake = []
        for folder in MANIPULATION_FOLDERS:
            folder_names = [n for n in all_names if f"/{folder}/" in n and n.endswith(".mp4")]
            selected_fake.extend(folder_names[:per_folder_count])
        selected_fake = selected_fake[:fake_count]

        print(f"[Video Preprocess] Extracting {len(selected_real)} Real videos...")
        for name in selected_real:
            target_file = real_dir / Path(name).name
            if not target_file.exists():
                with archive.open(name) as src, open(target_file, "wb") as dst:
                    dst.write(src.read())

        print(f"[Video Preprocess] Extracting {len(selected_fake)} Fake videos...")
        for name in selected_fake:
            parts = Path(name).parts
            category = parts[1] if len(parts) > 1 else "manipulated"
            cat_dir = fake_dir / category
            cat_dir.mkdir(parents=True, exist_ok=True)
            target_file = cat_dir / Path(name).name
            if not target_file.exists():
                with archive.open(name) as src, open(target_file, "wb") as dst:
                    dst.write(src.read())

    real_videos = list(real_dir.glob("*.mp4"))[:real_count]
    fake_videos = list(fake_dir.glob("**/*.mp4"))[:fake_count]
    print(f"[Video Preprocess] Extraction complete. Real: {len(real_videos)}, Fake: {len(fake_videos)}")
    return {"real": real_videos, "fake": fake_videos}


def extract_evenly_spaced_frames(video_path: Path, num_frames: int = 10) -> List[np.ndarray]:
    """
    Extracts num_frames evenly spaced frames from an MP4 video using OpenCV.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)

    cap.release()
    return frames


# MediaPipe Face Detector setup
_mp_face_detector = None
if MP_FACE_AVAILABLE:
    try:
        _mp_face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
    except Exception:
        _mp_face_detector = None

# OpenCV Haar Cascade fallback
_haar_cascade = None


def _get_haar_cascade():
    global _haar_cascade
    if _haar_cascade is None:
        xml_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _haar_cascade = cv2.CascadeClassifier(xml_path)
    return _haar_cascade


def detect_and_crop_face(frame_bgr: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
    """
    Detects the primary face using MediaPipe (or Haar cascade fallback),
    adds 15% margin, crops, resizes to target_size (224x224), and returns RGB image.
    Returns None if no face is detected.
    """
    h, w, _ = frame_bgr.shape

    # 1. Try MediaPipe
    if _mp_face_detector is not None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = _mp_face_detector.process(frame_rgb)
        if results.detections:
            best_det = max(
                results.detections,
                key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height,
            )
            bbox = best_det.location_data.relative_bounding_box
            xmin = int(bbox.xmin * w)
            ymin = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            # Add 15% margin
            pad_x = int(bw * 0.15)
            pad_y = int(bh * 0.15)
            x1 = max(0, xmin - pad_x)
            y1 = max(0, ymin - pad_y)
            x2 = min(w, xmin + bw + pad_x)
            y2 = min(h, ymin + bh + pad_y)

            if x2 > x1 and y2 > y1:
                crop = frame_rgb[y1:y2, x1:x2]
                return cv2.resize(crop, target_size)

    # 2. Haar Cascade fallback
    cascade = _get_haar_cascade()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    if len(faces) > 0:
        # Pick largest face
        x, y, bw, bh = max(faces, key=lambda f: f[2] * f[3])
        pad_x = int(bw * 0.15)
        pad_y = int(bh * 0.15)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        crop = frame_rgb[y1:y2, x1:x2]
        return cv2.resize(crop, target_size)

    # Ignore frames without face as specified
    return None


class VideoFaceDataset(Dataset):
    """
    Dataset representing videos where each item returns:
    (frames_tensor, label) -> frames_tensor shape: (N_FRAMES, 3, 224, 224), label: float
    """

    def __init__(self, video_dict: Dict[str, List[Path]], num_frames: int = 10, cache_dir: Path = CROPS_CACHE_DIR):
        self.num_frames = num_frames
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.samples = []  # List of (video_path, label)
        for v in video_dict.get("real", []):
            self.samples.append((v, 0.0))
        for v in video_dict.get("fake", []):
            self.samples.append((v, 1.0))

        random.seed(42)
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def _process_video(self, video_path: Path) -> torch.Tensor:
        # Cache key based on file stem
        cache_file = self.cache_dir / f"{video_path.stem}_crops.pt"
        if cache_file.exists():
            try:
                return torch.load(cache_file, weights_only=True)
            except Exception:
                pass

        raw_frames = extract_evenly_spaced_frames(video_path, num_frames=self.num_frames * 2)
        face_crops = []
        for frame in raw_frames:
            crop = detect_and_crop_face(frame)
            if crop is not None:
                face_crops.append(crop)
            if len(face_crops) == self.num_frames:
                break

        # If fewer than num_frames detected, duplicate last frame or pad with zeros
        while len(face_crops) < self.num_frames:
            if len(face_crops) > 0:
                face_crops.append(face_crops[-1])
            else:
                face_crops.append(np.zeros((224, 224, 3), dtype=np.uint8))

        tensor_list = [self.transform(Image.fromarray(c)) for c in face_crops]
        video_tensor = torch.stack(tensor_list)  # (10, 3, 224, 224)

        torch.save(video_tensor, cache_file)
        return video_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path, label = self.samples[idx]
        frames_tensor = self._process_video(video_path)
        return frames_tensor, torch.tensor(label, dtype=torch.float32)


def get_video_dataloaders(
    batch_size: int = 16,
    val_split: float = 0.2,
    real_count: int = 100,
    fake_count: int = 100,
) -> Tuple[DataLoader, DataLoader]:
    """
    Builds train and validation DataLoaders for video deepfake detection.
    """
    video_dict = extract_balanced_videos(real_count=real_count, fake_count=fake_count)
    full_ds = VideoFaceDataset(video_dict)

    val_size = int(len(full_ds) * val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available()
    )

    print(f"[Video Preprocess] Video dataset ready: {len(train_ds)} train videos, {len(val_ds)} val videos.")
    return train_loader, val_loader


if __name__ == "__main__":
    v_dict = extract_balanced_videos(real_count=2, fake_count=2)
    print("Extracted sample videos:", v_dict)
