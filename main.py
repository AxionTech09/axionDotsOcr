from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from PIL import Image

# Try PaddleOCR first; fallback to pytesseract if needed
try:
    from paddleocr import PaddleOCR
    ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')
    use_paddle = True
except ImportError:
    import pytesseract
    use_paddle = False

app = FastAPI(title="DOTS OCR CPU API", description="CPU-based OCR extraction (no vLLM required)", version="1.0")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"message": "DOTS OCR CPU-based API is running"}


@app.post("/api/ocr")
async def extract_text(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Open image
        image = Image.open(file_path)

        # OCR Processing
        if use_paddle:
            result = ocr_engine.ocr(file_path, cls=True)
            text_blocks = [line[1][0] for block in result for line in block]
        else:
            text_blocks = pytesseract.image_to_string(image).splitlines()

        extracted_text = "\n".join([t for t in text_blocks if t.strip()])

        return JSONResponse(content={"status": "success", "data": extracted_text})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
