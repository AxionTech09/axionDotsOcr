import sys, os, json
from pdf2image import convert_from_path

# ✅ Correct import for your repo version
from dots_ocr.parser import DotsOCRParser


def process_file(file_path):
    # Initialize OCR parser (no device arg)
    ocr = DotsOCRParser()

    results = []
    if file_path.lower().endswith(".pdf"):
        print("Processing PDF, converting pages...")
        pages = convert_from_path(file_path, dpi=200)
        for i, page in enumerate(pages):
            temp_img = f"page_{i+1}.png"
            page.save(temp_img, "PNG")
            print(f"Running OCR on page {i+1}...")
            res = ocr.parse(temp_img)
            results.append({"page": i + 1, "result": res})
            os.remove(temp_img)
    else:
        print("Processing single image...")
        res = ocr.parse(file_path)
        results.append({"page": 1, "result": res})

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_ocr.py <file_path>")
        sys.exit(1)
    process_file(sys.argv[1])
