import os
import json
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import argparse

from dots_ocr.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.image_utils import get_image_by_fitz_doc, fetch_image, smart_resize
from dots_ocr.utils.doc_utils import load_images_from_pdf
from dots_ocr.utils.prompts import dict_promptmode_to_prompt
from dots_ocr.utils.layout_utils import post_process_output, draw_layout_on_image, pre_process_bboxes
from dots_ocr.utils.format_transformer import layoutjson2md
from dots_ocr.utils.cpu_optimizer import CPUOptimizer, configure_cpu_inference
from qwen_vl_utils import process_vision_info


class DotsOCRParser:
    """CPU-only version of dots.OCR (Hugging-Face backend)."""

    def __init__(
        self,
        dpi=200,
        output_dir="./output",
        min_pixels=None,
        max_pixels=None,
        use_hf=True,
        model_path="/var/www/dots_ocr/dots_ocr/local_model",
    ):
        print("⚙️  Initializing dots.OCR in CPU mode (Hugging Face backend)...")
        configure_cpu_inference()

        self.dpi = dpi
        self.output_dir = output_dir
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.model_path = model_path
        self.use_hf = use_hf

        self._load_hf_model()

        assert self.min_pixels is None or self.min_pixels >= MIN_PIXELS
        assert self.max_pixels is None or self.max_pixels <= MAX_PIXELS

    # ------------------------------------------------------------------ #
    #  MODEL  LOADING
    # ------------------------------------------------------------------ #
    def _load_hf_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        print("🔹 Loading model from local path:", self.model_path)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )

            # ✅ Deep-convert every param & buffer to float32
            for name, p in self.model.named_parameters():
                if p.dtype != torch.float32:
                    p.data = p.data.float()
            for name, b in self.model.named_buffers():
                if b.dtype != torch.float32 and b.dtype not in [
                    torch.long,
                    torch.int,
                    torch.bool,
                ]:
                    b.data = b.data.float()

            # ✅ Apply any optional CPU optimizations
            self.model = CPUOptimizer.optimize_model_for_cpu(self.model)

            # ✅ Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )

            print("✅ Model successfully loaded and optimized for CPU")
            print(f"   Model device: {next(self.model.parameters()).device}")
            print(f"   Model dtype: {next(self.model.parameters()).dtype}")

            mem = CPUOptimizer.get_memory_info()
            print(
                f"💾 Memory usage: {mem['cpu_memory_used_gb']:.1f}GB / {mem['cpu_memory_total_gb']:.1f}GB"
            )

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔧 Attempting fallback loading method...")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map={"": "cpu"},
                low_cpu_mem_usage=True,
                use_safetensors=False,
            ).float()

            # Force convert deeply
            self._fix_model_dtypes()

            self.processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            print("✅ Model loaded using fallback method")

    # ------------------------------------------------------------------ #
    #  INFERENCE
    # ------------------------------------------------------------------ #
    def _inference_with_hf(self, image, prompt):
        import torch

        torch.set_default_dtype(torch.float32)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs_cpu = CPUOptimizer.prepare_inputs_for_cpu(inputs)

        with torch.no_grad():
            self.model = self.model.float()
            try:
                generated_ids = self.model.generate(
                    **inputs_cpu,
                    max_new_tokens=2048,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            except RuntimeError as e:
                if "BFloat16" in str(e) or "bias type" in str(e):
                    print("⚠️  Detected dtype mismatch. Attempting model conversion...")
                    self._fix_model_dtypes()
                    generated_ids = self.model.generate(
                        **inputs_cpu,
                        max_new_tokens=2048,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                    )
                else:
                    raise e

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(
                    inputs_cpu["input_ids"], generated_ids
                )
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        return response

    # ------------------------------------------------------------------ #
    #  DTYPE  FIX
    # ------------------------------------------------------------------ #
    def _fix_model_dtypes(self):
        """Force-convert all nested parameters/buffers to float32."""
        import torch

        print("🔧 Converting all model parameters to float32...")

        def convert_to_float32(module):
            for child in module.children():
                convert_to_float32(child)
            for _, p in module.named_parameters(recurse=False):
                if p.dtype != torch.float32:
                    p.data = p.data.float()
            for _, b in module.named_buffers(recurse=False):
                if b.dtype != torch.float32 and b.dtype not in [
                    torch.long,
                    torch.int,
                    torch.bool,
                ]:
                    b.data = b.data.float()

        convert_to_float32(self.model)
        print("✅ Model dtype conversion completed")

    # ------------------------------------------------------------------ #
    #  PROMPT / PARSE HELPERS
    # ------------------------------------------------------------------ #
    def get_prompt(self, prompt_mode, bbox=None, origin_image=None, image=None, min_pixels=None, max_pixels=None):
        prompt = dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == "prompt_grounding_ocr":
            assert bbox is not None
            bboxes = [bbox]
            bbox = pre_process_bboxes(
                origin_image,
                bboxes,
                input_width=image.width,
                input_height=image.height,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )[0]
            prompt = prompt + str(bbox)
        return prompt

    def _parse_single_image(
        self,
        origin_image,
        prompt_mode,
        save_dir,
        save_name,
        source="image",
        page_idx=0,
        bbox=None,
        fitz_preprocess=False,
    ):
        min_pixels, max_pixels = self.min_pixels, self.max_pixels
        if prompt_mode == "prompt_grounding_ocr":
            min_pixels = min_pixels or MIN_PIXELS
            max_pixels = max_pixels or MAX_PIXELS

        if source == "image" and fitz_preprocess:
            image = get_image_by_fitz_doc(origin_image, target_dpi=self.dpi)
            image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            image = fetch_image(origin_image, min_pixels=min_pixels, max_pixels=max_pixels)

        input_height, input_width = smart_resize(image.height, image.width)
        prompt = self.get_prompt(
            prompt_mode, bbox, origin_image, image, min_pixels=min_pixels, max_pixels=max_pixels
        )

        response = self._inference_with_hf(image, prompt)
        result = {"page_no": page_idx, "input_height": input_height, "input_width": input_width}

        if source == "pdf":
            save_name = f"{save_name}_page_{page_idx}"

        if prompt_mode in ["prompt_layout_all_en", "prompt_layout_only_en", "prompt_grounding_ocr"]:
            cells, _ = post_process_output(
                response, prompt_mode, origin_image, image, min_pixels=min_pixels, max_pixels=max_pixels
            )

            json_file_path = os.path.join(save_dir, f"{save_name}.json")
            with open(json_file_path, "w", encoding="utf-8") as w:
                json.dump(cells, w, ensure_ascii=False)

            image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
            origin_image.save(image_layout_path)

            md_content = layoutjson2md(origin_image, cells, text_key="text")
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(md_content)

            result.update(
                {
                    "layout_info_path": json_file_path,
                    "layout_image_path": image_layout_path,
                    "md_content_path": md_file_path,
                }
            )
        else:
            md_content = response
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(md_content)
            result.update({"md_content_path": md_file_path})

        return result

    # ------------------------------------------------------------------ #
    #  FILE  ENTRYPOINTS
    # ------------------------------------------------------------------ #
    def parse_image(self, input_path, filename, prompt_mode, save_dir, bbox=None, fitz_preprocess=False):
        origin_image = fetch_image(input_path)
        result = self._parse_single_image(
            origin_image, prompt_mode, save_dir, filename, source="image", bbox=bbox, fitz_preprocess=fitz_preprocess
        )
        result["file_path"] = input_path
        return [result]

    def parse_pdf(self, input_path, filename, prompt_mode, save_dir):
        print(f"Loading PDF: {input_path}")
        images_origin = load_images_from_pdf(input_path, dpi=self.dpi)
        total_pages = len(images_origin)
        tasks = [
            {
                "origin_image": img,
                "prompt_mode": prompt_mode,
                "save_dir": save_dir,
                "save_name": filename,
                "source": "pdf",
                "page_idx": i,
            }
            for i, img in enumerate(images_origin)
        ]

        def _execute(task_args):
            return self._parse_single_image(**task_args)

        print(f"Parsing PDF with {total_pages} pages on CPU...")
        results = []
        with ThreadPool(1) as pool:
            with tqdm(total=total_pages, desc="Processing PDF pages") as pbar:
                for r in pool.imap_unordered(_execute, tasks):
                    results.append(r)
                    pbar.update(1)
        results.sort(key=lambda x: x["page_no"])
        for r in results:
            r["file_path"] = input_path
        return results

    def parse_file(self, input_path, output_dir="", prompt_mode="prompt_layout_all_en", bbox=None, fitz_preprocess=False):
        output_dir = output_dir or self.output_dir
        output_dir = os.path.abspath(output_dir)
        filename, file_ext = os.path.splitext(os.path.basename(input_path))
        save_dir = os.path.join(output_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

        if file_ext == ".pdf":
            results = self.parse_pdf(input_path, filename, prompt_mode, save_dir)
        elif file_ext in image_extensions:
            results = self.parse_image(input_path, filename, prompt_mode, save_dir, bbox=bbox, fitz_preprocess=fitz_preprocess)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        print(f"Parsing finished. Results saved to {save_dir}")
        with open(os.path.join(output_dir, f"{filename}.jsonl"), "w", encoding="utf-8") as w:
            for r in results:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        return results
