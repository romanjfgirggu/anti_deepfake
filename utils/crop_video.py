import argparse
import json
import random
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm
from dataclasses import dataclass
import shutil




@dataclass
class PreparationVideo:
    VIDEO_EXTS = {".mp4"}
    IMG_EXTS = {".jpg", ".jpeg", ".png"}

    def _list_d_video(self, video_path: str) -> list:
        return [p for p in Path(video_path).rglob("*") if p.suffix.lower() in self.VIDEO_EXTS]

    def _list_d_img(self, video_path: str) -> list:
        return [p for p in Path(video_path).rglob("*") if p.suffix.lower() in self.IMG_EXTS]


    def _align_face_by_eyes(self, face_rgb, landmarks, out_size=256, eye_y=0.38, eye_dist_x=0.35):
        left_eye = landmarks[0].astype(np.float32)
        right_eye = landmarks[1].astype(np.float32)

        eyes_center = (left_eye + right_eye) / 2.0
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]

        angle = np.degrees(np.arctan2(dy, dx))

        dist = np.sqrt(dx * dx + dy * dy)
        dist = max(dist, 1e-6)

        desired_dist = out_size * eye_dist_x

        scale = desired_dist / dist

        M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale)

        target_eye_center = np.array([out_size * 0.5, out_size * eye_y], dtype=np.float32)

        M[0, 2] += target_eye_center[0] - eyes_center[0]
        M[1, 2] += target_eye_center[1] - eyes_center[1]

        aligned = cv2.warpAffine(
            face_rgb,
            M,
            (out_size, out_size),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return aligned, M

    def _expand_box(self, box, w, h, margin=0.35):
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        x1 = max(0, int(x1 - bw * margin))
        y1 = max(0, int(y1 - bh * margin))
        x2 = min(w, int(x2 + bw * margin))
        y2 = min(h, int(y2 + bh * margin))

        return x1, y1, x2, y2

    def _extract_n_frames(self, video_path: str, n_frames: int = 5) -> list:
        frames = []

        video_path = Path(video_path)

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Не удалось прочитать кадры из {video_path}")

        frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                continue
            frames.append(frame)
        cap.release()
        return frames

    def _save_image(self, path: Path, rgb_img: np.ndarray):
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)

    def fase_crop(self, videos_path: str, out_images_path: str) -> None:
        mtcnn = MTCNN(keep_all=True, device="cuda" if torch.cuda.is_available() else "cpu")
        videos_path = self._list_d_video(videos_path)
        for video_path in tqdm(videos_path, desc="Processing"):
            stem = video_path.stem
            count_frames = 0

            frames = self._extract_n_frames(video_path, n_frames=20)
            for frame in frames:
                h, w = frame.shape[:2]

                boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)
                if boxes is None or probs is None or landmarks is None:
                    continue

                best_idx = int(np.argmax(probs))
                box = boxes[best_idx]
                lmk = landmarks[best_idx]

                x1, y1, x2, y2 = self._expand_box(box, w, h)
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                lmk_local = lmk.copy()
                lmk_local[:, 0] -= x1
                lmk_local[:, 1] -= y1

                aligned_face, M = self._align_face_by_eyes(
                    face_rgb=face_crop,
                    landmarks=lmk_local,
                    out_size=256,
                )
                aligned_out = Path(out_images_path) / f"{stem}_frame_{count_frames}.png"
                count_frames += 1
                self._save_image(aligned_out, aligned_face)

        print("Готово!")

