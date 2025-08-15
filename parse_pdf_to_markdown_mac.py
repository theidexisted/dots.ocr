

"""
Parses a full PDF into a single Markdown file on macOS/M1.
This script runs the model locally on the MPS device and does not require a server.
"""

import os
import argparse
import json
import gc
import fitz  # PyMuPDF
from PIL import Image
import torch
import sys
import types
import re
import importlib.machinery as _machinery
from tqdm import tqdm

# --- Utility Imports from the project ---
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from dots_ocr.utils.prompts import dict_promptmode_to_prompt
from dots_ocr.utils.format_transformer import layoutjson2md

# --- Shim from m1.py to avoid flash_attn errors -----------------------------
def setup_flash_attn_shim():
    if "flash_attn" in sys.modules:
        return
    _m = types.ModuleType("flash_attn")
    _m.__spec__ = _machinery.ModuleSpec(name="flash_attn", loader=None)
    _m.__path__ = []
    def flash_attn_varlen_func(*args, **kwargs):
        raise RuntimeError("flash_attn is not available.")
    _m.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["flash_attn"] = _m

setup_flash_attn_shim()

# --- Core OCR Class (adapted from mac/demo_gradio_m1.py) --------------------
class DotsOCRM1:
    def __init__(self, model_dir="./weights/DotsOCR"):
        self.model_dir = model_dir
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "mps" else torch.float32
        self.model = None
        self.processor = None
        self.init_model()

    def get_model_config(self):
        config = AutoConfig.from_pretrained(self.model_dir, trust_remote_code=True, local_files_only=True)
        if getattr(config, "vision_config", None) is not None:
            config.vision_config.attn_implementation = "sdpa"
        else:
            setattr(config, "attn_implementation", "sdpa")
        if hasattr(config, "use_sliding_window"):
            config.use_sliding_window = False
        if hasattr(config, "sliding_window"):
            config.sliding_window = None
        return config

    def patch_vision_tower(self):
        orig_forward_func = self.model.vision_tower.forward.__func__
        def _forward_no_bf16(self, hidden_states, grid_thw, bf16=True):
            return orig_forward_func(self, hidden_states, grid_thw, bf16=False)
        self.model.vision_tower.forward = types.MethodType(_forward_no_bf16, self.model.vision_tower)

    def init_model(self):
        if self.model is not None:
            return
        print("Initializing M1-optimized model...")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        min_pix, max_pix = 256 * 28 * 28 * 4, 640 * 28 * 28 * 4
        config = self.get_model_config()
        self.processor = AutoProcessor.from_pretrained(self.model_dir, local_files_only=True, min_pixels=min_pix, max_pixels=max_pix, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, local_files_only=True, trust_remote_code=True, config=config, attn_implementation="sdpa", torch_dtype=self.dtype, low_cpu_mem_usage=True)
        self.model.to(self.device)
        if self.device == "mps":
            self.patch_vision_tower()
        print(f"[info] Model loaded on device={self.device}, dtype={self.dtype}")

    def prepare_inputs(self, img, prompt_text):
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt_text}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                v = v.to(self.device)
                if torch.is_floating_point(v):
                    v = v.to(dtype=self.dtype)
                inputs[k] = v
        return inputs

    def run_inference(self, inputs, max_new_tokens=1536):
        with torch.inference_mode():
            return self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=None, do_sample=False)

    def decode_output(self, out_ids, inputs):
        trimmed_ids = [o[len(iids):] for iids, o in zip(inputs["input_ids"], out_ids)]
        return self.processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def parse_json_flex(self, s: str):
        s = s.strip()
        if s.startswith("{") and s.endswith("}"): return json.loads(s)
        if s.startswith("[") and s.endswith("]"): return json.loads(s)
        m_obj = re.search(r"\{[\s\S]*\}", s)
        m_arr = re.search(r"\[[\s\S]*\]", s)
        m = m_obj if m_obj and (not m_arr or m_obj.start() < m_arr.start()) else m_arr
        if not m: raise ValueError("No JSON object or array found.")
        return json.loads(m.group(0))

    def run(self, image, prompt_mode):
        prompt_text = dict_promptmode_to_prompt.get(prompt_mode, "")
        inputs = self.prepare_inputs(image, prompt_text)
        out_ids = self.run_inference(inputs)
        out_text = self.decode_output(out_ids, inputs)
        del inputs, out_ids
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        return out_text

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Parse a PDF to a single Markdown file using the local M1-optimized model.")
    parser.add_argument("input_pdf", help="The absolute path to the input PDF file.")
    parser.add_argument("--output_dir", default="./output", help="Directory to save the final markdown file.")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for rendering PDF pages.")
    args = parser.parse_args()

    # --- 1. Initialize the M1-specific OCR model ---
    ocr_model = DotsOCRM1()

    # --- 2. Load PDF and process each page ---
    print(f"Loading PDF from: {args.input_pdf}")
    doc = fitz.open(args.input_pdf)
    full_markdown_content = []

    for page_num in tqdm(range(len(doc)), desc="Processing PDF pages"):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=args.dpi)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Run inference to get raw text output (which should be JSON)
        raw_output = ocr_model.run(image, prompt_mode="prompt_layout_all_en")

        try:
            # Parse the JSON output
            json_output = ocr_model.parse_json_flex(raw_output)
            # Convert the layout JSON to Markdown
            md_content = layoutjson2md(image, json_output, text_key='text')
            full_markdown_content.append(md_content)
        except (ValueError, TypeError) as e:
            print(f"\nCould not process page {page_num + 1}: {e}")
            print("Falling back to including raw output for this page.")
            full_markdown_content.append(f"<!-- Page {page_num + 1}: Error processing JSON. Raw output below. -->\n{raw_output}")
        finally:
            del image, raw_output
            gc.collect()

    # --- 3. Combine and save the final Markdown file ---
    os.makedirs(args.output_dir, exist_ok=True)
    pdf_filename = os.path.splitext(os.path.basename(args.input_pdf))[0]
    combined_md_path = os.path.join(args.output_dir, f"{pdf_filename}_full_mac.md")

    with open(combined_md_path, 'w', encoding='utf-8') as f:
        f.write('\n\n---\n\n'.join(full_markdown_content))
    
    print(f"\nSuccessfully created combined markdown file at: {combined_md_path}")

if __name__ == "__main__":
    main()
