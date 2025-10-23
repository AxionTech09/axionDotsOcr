from dotsocr import DotsOCR
from pdf2image import convert_from_path
import sys, os, json

def process_file(file_path):
    ocr = DotsOCR(model_name="dots-ocr-small", device="cpu")  # Load model on CPU

    results = []
    if file_path.lower().endswith(".pdf"):
        pages = convert_from_path(file_path, dpi=200)
        for i, page in enumerate(pages):
            temp_img = f"page_{i+1}.png"
            page.save(temp_img, "PNG")
            res = ocr(temp_img)
            results.append({"page": i + 1, "result": res})
            os.remove(temp_img)
    else:
        res = ocr(file_path)
        results.append({"page": 1, "result": res})

    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_ocr.py <file_path>")
        sys.exit(1)

    process_file(sys.argv[1])
