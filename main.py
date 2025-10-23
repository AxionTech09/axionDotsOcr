from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from PIL import Image

# Import the OCR function from the repo
from dots_ocr.model.inference import inference_with_vllm

app = FastAPI(title="DOTS OCR API", description="OCR Extraction for Vehicle Insurance Documents", version="1.0")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"message": "DOTS OCR API is running"}


@app.post("/api/ocr")
async def extract_text(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Open image using Pillow
        image = Image.open(file_path)

        # Simple OCR prompt for extraction
        prompt = "Extract all text content clearly from the image."

        # Call the inference function
        result = inference_with_vllm(
            image=image,
            prompt=prompt,
            protocol="http",
            ip="localhost",
            port=8000,
            model_name="rednote-hilab/dots.ocr"
        )

        if not result:
            return JSONResponse(content={"status": "error", "message": "OCR failed or returned empty."}, status_code=500)

        return JSONResponse(content={"status": "success", "data": result})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
