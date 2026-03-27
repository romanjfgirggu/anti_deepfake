from pathlib import Path
from PIL import Image
import cv2
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
from dataclasses import dataclass

from tqdm import tqdm

from utils.crop_video import PreparationVideo

from utils.image_processing import Embeddings
from sklearn.metrics import roc_auc_score

import joblib

@dataclass
class Model:
    def _iter_image_batches(self, image_paths, batch_size=32):
        batch = []

        for image_path in image_paths:
            with Image.open(image_path) as image:
                image = image.convert("RGB").resize((256, 256))
                image_array = np.asarray(image, dtype=np.float32)
                batch.append(image_array)

            if len(batch) == batch_size:
                yield np.stack(batch, axis=0)
                batch = []

        if batch:
            yield np.stack(batch, axis=0)

    def _load_images_from_dir(self, directory: str, label: int, emb: Embeddings, batch_size=32):
        x_batches = []
        y_batches = []

        photo_dir = Path(directory) / "photo"
        image_paths = [
            p for p in photo_dir.iterdir()
            if p.is_file() and p.suffix.lower() in PreparationVideo.IMG_EXTS
        ]

        for batch in tqdm(self._iter_image_batches(image_paths, batch_size=batch_size), total=(len(image_paths) + batch_size - 1) // batch_size):
            embs = emb.extract_embeddings_batch(batch)
            x_batches.append(embs)
            y_batches.append(np.full(len(embs), label, dtype=np.int32))

        x = np.vstack(x_batches) if x_batches else np.empty((0, 2048), dtype=np.float32)
        y = np.concatenate(y_batches) if y_batches else np.empty((0,), dtype=np.int32)
        return x, y

    def _dataset_create(self):
        emb = Embeddings(batch_size=32)

        x_real, y_real = self._load_images_from_dir(r"dataset\real_media", 0, emb)
        x_fake, y_fake = self._load_images_from_dir(r"dataset\fake_media", 1, emb)

        x = np.vstack([x_real, x_fake])
        y = np.concatenate([y_real, y_fake])
        return x, y

    def traning_model(self) -> Pipeline:
        x, y = self._dataset_create()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("PCA", PCA()),
            ("svc", SVC(C=1.0, class_weight="balanced", max_iter=10000, probability=True)),
        ])
        clf.fit(x_train, y_train)
        print(f"ROC-AUC: {roc_auc_score(y_test, clf.predict_proba(x_test)[:, -1], multi_class='ovr')}")
        return clf

    def save_model(self, file_name: str = "model", model: Pipeline | None = None):
        if not model:
            raise "Модель не обучена"
        else:
            joblib.dump(model, f"models\\{file_name}.joblib")

