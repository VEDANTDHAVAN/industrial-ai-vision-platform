from io import BytesIO

from fastapi import FastAPI, UploadFile, File
from PIL import Image

from src.api.inference import predict
from src.api.schemas import PredictionResponse
from src.api.logger import log_prediction

app = FastAPI(
    title="Industrial AI Vision Platform"
)

@app.get("/")
def root():
    return {
        "message": "Casting Defect Detection API"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    result = predict(image)
    
    assert file.filename is not None
    
    log_prediction(
        filename= file.filename, prediction= result["prediction"], confidence= result["confidence"],
        ok_probability= result["ok_probability"], defect_probability= result["defect_probability"], model_version= result["model_version"]
    )

    return result 