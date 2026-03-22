import io
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn


CLASSES = ["Train", "Bee", "Castle"]
IMG_SIZE = 128
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    try:
        import torch_directml  # type: ignore

        return torch_directml.device()
    except Exception:
        return torch.device("cpu")


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def resolve_model_path() -> Path:
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"MODEL_PATH does not exist: {p}")

    candidates = [
        Path("models/simple_cnn_classifier.pth"),
        Path("models/simple_cnn_classifier.safetensors"),
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find model weights. Expected one of: "
        "models/simple_cnn_classifier.pth, models/simple_cnn_classifier.safetensors"
    )


def load_state_dict(weights_path: Path) -> Dict[str, torch.Tensor]:
    if weights_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "safetensors file found but safetensors package is not installed."
            ) from exc

        return load_file(str(weights_path))

    payload = torch.load(str(weights_path), map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload

    raise RuntimeError("Unsupported checkpoint format.")


def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)


def predict_from_image(image: Image.Image) -> Dict[str, object]:
    x = preprocess_image(image).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = CLASSES[pred_idx]
    probabilities = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

    return {
        "predicted_class": pred_class,
        "confidence": float(probs[pred_idx]),
        "probabilities": probabilities,
    }


async def read_uploaded_image(file: UploadFile) -> Image.Image:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"{file.filename}: upload an image file.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"{file.filename}: uploaded file is empty.")

    try:
        return Image.open(io.BytesIO(data))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail=f"{file.filename}: could not decode image.") from exc


app = FastAPI(
    title="Simple CNN Image Classifier API",
    version="1.0.0",
    description="Predicts one of: Train, Bee, Castle",
)

DEVICE = get_device()
MODEL_PATH = resolve_model_path()
MODEL = SimpleCNN(num_classes=len(CLASSES))
MODEL.load_state_dict(load_state_dict(MODEL_PATH), strict=True)
MODEL = MODEL.to(DEVICE)
MODEL.eval()


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, object]:
    image = await read_uploaded_image(file)
    return predict_from_image(image)


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)) -> Dict[str, object]:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > 32:
        raise HTTPException(status_code=400, detail="Too many files. Maximum is 32 per request.")

    results = []
    for file in files:
        image = await read_uploaded_image(file)
        prediction = predict_from_image(image)
        results.append(
            {
                "filename": file.filename,
                **prediction,
            }
        )

    return {
        "count": len(results),
        "results": results,
    }
