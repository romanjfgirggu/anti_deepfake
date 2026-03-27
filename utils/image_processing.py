# image_processing.py
import numpy as np
import tensorflow as tf


class Embeddings:
    def __init__(self, img_size=(256, 256), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.extractor = self._build_extractor(img_size)

    def _build_extractor(self, img_size=(256, 256)):
        return tf.keras.applications.Xception(
            include_top=False,
            weights="imagenet",
            input_shape=(img_size[0], img_size[1], 3),
            pooling="avg",
        )

    def extract_embeddings_batch(self, images: np.ndarray):
        images = tf.keras.applications.xception.preprocess_input(images)
        return self.extractor.predict(images, batch_size=self.batch_size, verbose=0)