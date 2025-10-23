import sys, os, json
from pdf2image import convert_from_path
from dots_ocr.parser import DotsOCRParser

def process_file(file_path):
    ocr = DotsOCRParser()

    results = []
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    if file_path.lower().endswith(".pdf"):
        print(f"Processing PDF: {file_path}")
        # The parser has a built-in parse_pdf method
        res = ocr.parse_pdf(
            input_path=file_path,
            filename=base_name,
            prompt_mode="ocr",
            save_dir=output_dir
        )
        results.append({"file": base_name, "result": res})
    else:
        print(f"Processing image: {file_path}")
        # Use parse_image for single images
        res = ocr.parse_image(
            input_path=file_path,
            filename=base_name,
            prompt_mode="ocr",
            save_dir=output_dir
        )
        results.append({"file": base_name, "result": res})

    # Save JSON output
    output_path = os.path.join(output_dir, f"{base_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ OCR completed successfully!")
    print(f"📄 Output saved to: {output_path}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_ocr.py <file_path>")
        sys.exit(1)
    process_file(sys.argv[1])
