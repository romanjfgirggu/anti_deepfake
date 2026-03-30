from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from tqdm import tqdm

from api.health import router as health_router
import uvicorn
from utils.crop_video import PreparationVideo
from utils.image_processing import Embeddings


def iter_image_batches(image_paths, batch_size=32):
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




app = FastAPI(title="Anti-Deepfake")
app.include_router(health_router)

@app.post("/v1/detect/video")
async def detect_video(file: UploadFile | None = File(default=None)):
    x_batches = []
    if file is None:
        raise HTTPException(status_code=400, detail="Provide file or video_url")

    file_path = Path("../temp/video") / file.filename
    with open(file_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            buffer.write(chunk)


    pv = PreparationVideo()

    emb = Embeddings()
    pv.fase_crop("../temp/video", "../temp/photos")

    for batch in tqdm(iter_image_batches([p for p in Path("../temp/photos").iterdir() if p.is_file() and p.suffix.lower() in PreparationVideo.IMG_EXTS], batch_size=32),
                      total=(len([p for p in Path("../temp/photos").iterdir() if p.is_file() and p.suffix.lower() in PreparationVideo.IMG_EXTS]) + 32 - 1) // 32):
        embs = emb.extract_embeddings_batch(batch)
        x_batches.append(embs)
    x = np.vstack(x_batches) if x_batches else np.empty((0, 2048), dtype=np.float32)
    model = joblib.load("../models/model.joblib")
    predict = model.predict(x)
    await file.close()
    predict = predict.sum() / len(predict)
    for folder in [Path("../temp/photos"), Path("../temp/video")]:
        for item in folder.iterdir():
            if item.is_file():
                item.unlink()
    return {"Predicted": "fake" if predict > 0.7 else "real",
            "Confidence": 1-predict}




if __name__ == '__main__':
    uvicorn.run("main:app")
