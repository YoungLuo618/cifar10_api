# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
from models.cnn import CIFAR10CNN
from infer_utils import load_image_to_tensor, logits_to_pred, CIFAR10_CLASSES

app = FastAPI(
    title="CIFAR10 Classifier API",
    version="1.0.0",
    description="Upload an image to /predict to get a CIFAR-10 class prediction."
)

# ----- Model setup -----
DEVICE = torch.device("cpu")
MODEL = CIFAR10CNN().to(DEVICE)

WEIGHTS_PATH = "model.pth"
try:
    MODEL.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    MODEL.eval()
    MODEL_READY = True
except Exception as e:
    print(f"⚠️ Could not load weights '{WEIGHTS_PATH}': {e}")
    MODEL_READY = False


# ----- Basic routes -----
@app.get("/")
def root():
    """
    Welcome page. Visit /docs for Swagger UI.
    """
    return {
        "ok": True,
        "msg": "CIFAR10 API running. Go to /docs to test /predict.",
        "model_loaded": MODEL_READY
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_READY}


# ----- Prediction route -----
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file (jpg/png).")

    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model is not loaded. Run `python train.py` and restart the server.")

    try:
        img_bytes = await file.read()
        x = load_image_to_tensor(img_bytes).to(DEVICE)

        with torch.no_grad():
            logits = MODEL(x)
        idx, conf, probs = logits_to_pred(logits)

        return JSONResponse({
            "pred_index": idx,
            "pred_class": CIFAR10_CLASSES[idx],
            "confidence": round(conf, 4),
            "classes": CIFAR10_CLASSES  
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

