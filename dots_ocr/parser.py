import os
import json
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import argparse

# Removed vllm dependency for CPU-only
# from dots_ocr.model.inference import inference_with_vllm

from dots_ocr.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.image_utils import get_image_by_fitz_doc, fetch_image, smart_resize
from dots_ocr.utils.doc_utils import load_images_from_pdf
from dots_ocr.utils.prompts import dict_promptmode_to_prompt
from dots_ocr.utils.layout_utils import post_process_output, draw_layout_on_image, pre_process_bboxes
from dots_ocr.utils.format_transformer import layoutjson2md


class DotsOCRParser:
    """
    CPU-only version of DotsOCR parser.
    Runs on Hugging Face transformer backend (no vLLM server required).
    """

    def __init__(self,
                 dpi=200,
                 output_dir="./output",
                 min_pixels=None,
                 max_pixels=None,
                 use_hf=True,
                 model_path="rednote-hilab/dots-ocr-base"):
        self.dpi = dpi
        self.output_dir = output_dir
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.model_path = model_path

        self.use_hf = True
        print("⚙️  Initializing dots.OCR in CPU mode (Hugging Face backend)...")
        self._load_hf_model()

        assert self.min_pixels is None or self.min_pixels >= MIN_PIXELS
        assert self.max_pixels is None or self.max_pixels <= MAX_PIXELS

    def _load_hf_model(self):
        import os
        os.environ["HF_TOKEN"] = "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"
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from qwen_vl_utils import process_vision_info

        print("🔹 Loading model from:", self.model_path)
        device = "cpu"  # ✅ CPU-only

        self.model = AutoModelForCausalLM.from_pretrained(
            "rednote-hilab/dots-ocr-base",
            token=os.environ.get("HF_TOKEN"),
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            "rednote-hilab/dots-ocr-base",
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=True,
            use_fast=True
        )
        self.process_vision_info = process_vision_info

        print("✅ Model loaded successfully on CPU.")

    def _inference_with_hf(self, image, prompt):
        import torch
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cpu")

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        return response

    def get_prompt(self, prompt_mode, bbox=None, origin_image=None, image=None, min_pixels=None, max_pixels=None):
        prompt = dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == 'prompt_grounding_ocr':
            assert bbox is not None
            bboxes = [bbox]
            bbox = pre_process_bboxes(origin_image, bboxes,
                                      input_width=image.width, input_height=image.height,
                                      min_pixels=min_pixels, max_pixels=max_pixels)[0]
            prompt = prompt + str(bbox)
        return prompt

    def _parse_single_image(self,
                            origin_image,
                            prompt_mode,
                            save_dir,
                            save_name,
                            source="image",
                            page_idx=0,
                            bbox=None,
                            fitz_preprocess=False):
        min_pixels, max_pixels = self.min_pixels, self.max_pixels
        if prompt_mode == "prompt_grounding_ocr":
            min_pixels = min_pixels or MIN_PIXELS
            max_pixels = max_pixels or MAX_PIXELS

        if source == 'image' and fitz_preprocess:
            image = get_image_by_fitz_doc(origin_image, target_dpi=self.dpi)
            image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            image = fetch_image(origin_image, min_pixels=min_pixels, max_pixels=max_pixels)

        input_height, input_width = smart_resize(image.height, image.width)
        prompt = self.get_prompt(prompt_mode, bbox, origin_image, image, min_pixels=min_pixels, max_pixels=max_pixels)

        # ✅ Always use Hugging Face backend
        response = self._inference_with_hf(image, prompt)

        result = {'page_no': page_idx,
                  "input_height": input_height,
                  "input_width": input_width}

        if source == 'pdf':
            save_name = f"{save_name}_page_{page_idx}"

        if prompt_mode in ['prompt_layout_all_en', 'prompt_layout_only_en', 'prompt_grounding_ocr']:
            cells, filtered = post_process_output(response, prompt_mode, origin_image, image,
                                                  min_pixels=min_pixels, max_pixels=max_pixels)

            json_file_path = os.path.join(save_dir, f"{save_name}.json")
            with open(json_file_path, 'w', encoding="utf-8") as w:
                json.dump(cells, w, ensure_ascii=False)

            image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
            origin_image.save(image_layout_path)
            result.update({
                'layout_info_path': json_file_path,
                'layout_image_path': image_layout_path,
            })

            md_content = layoutjson2md(origin_image, cells, text_key='text')
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(md_content)
            result.update({'md_content_path': md_file_path})
        else:
            md_content = response
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(md_content)
            result.update({'md_content_path': md_file_path})

        return result

    def parse_image(self, input_path, filename, prompt_mode, save_dir, bbox=None, fitz_preprocess=False):
        origin_image = fetch_image(input_path)
        result = self._parse_single_image(origin_image, prompt_mode, save_dir, filename, source="image",
                                          bbox=bbox, fitz_preprocess=fitz_preprocess)
        result['file_path'] = input_path
        return [result]

    def parse_pdf(self, input_path, filename, prompt_mode, save_dir):
        print(f"Loading PDF: {input_path}")
        images_origin = load_images_from_pdf(input_path, dpi=self.dpi)
        total_pages = len(images_origin)
        tasks = [
            {
                "origin_image": image,
                "prompt_mode": prompt_mode,
                "save_dir": save_dir,
                "save_name": filename,
                "source": "pdf",
                "page_idx": i,
            } for i, image in enumerate(images_origin)
        ]

        def _execute_task(task_args):
            return self._parse_single_image(**task_args)

        num_thread = 1  # CPU mode = single-thread
        print(f"Parsing PDF with {total_pages} pages on CPU...")

        results = []
        with ThreadPool(num_thread) as pool:
            with tqdm(total=total_pages, desc="Processing PDF pages") as pbar:
                for result in pool.imap_unordered(_execute_task, tasks):
                    results.append(result)
                    pbar.update(1)

        results.sort(key=lambda x: x["page_no"])
        for r in results:
            r['file_path'] = input_path
        return results

    def parse_file(self, input_path, output_dir="", prompt_mode="prompt_layout_all_en", bbox=None,
                   fitz_preprocess=False):
        output_dir = output_dir or self.output_dir
        output_dir = os.path.abspath(output_dir)
        filename, file_ext = os.path.splitext(os.path.basename(input_path))
        save_dir = os.path.join(output_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

        if file_ext == '.pdf':
            results = self.parse_pdf(input_path, filename, prompt_mode, save_dir)
        elif file_ext in image_extensions:
            results = self.parse_image(input_path, filename, prompt_mode, save_dir, bbox=bbox,
                                       fitz_preprocess=fitz_preprocess)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        print(f"Parsing finished. Results saved to {save_dir}")
        with open(os.path.join(output_dir, f"{filename}.jsonl"), 'w', encoding="utf-8") as w:
            for result in results:
                w.write(json.dumps(result, ensure_ascii=False) + '\n')

        return results
