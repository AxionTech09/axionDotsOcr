from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os

# Import your OCR inference logic
from dots_ocr.model.inference import run_ocr  # adjust if the function name is different

app = FastAPI(title="DOTS OCR API", description="OCR Extraction for Vehicle Insurance Documents", version="1.0")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"message": "DOTS OCR API is running"}

@app.post("/api/ocr")
async def extract_text(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run the OCR function
        result = run_ocr(file_path)  # <-- replace this with the correct inference function

        return JSONResponse(content={"status": "success", "data": result})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
