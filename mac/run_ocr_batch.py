#!/usr/bin/env python
# -*- coding: utf-8 -*-"""

"""
Command-line batch OCR processing script based on the dots.ocr model.
Processes a single file (PDF/image) or a directory of images.
"""

import sys
import os
import argparse
from tqdm import tqdm
import glob
from typing import List, Tuple

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import io
import base64
import re
from pathlib import Path
from PIL import Image
import shutil
import types
import importlib.machinery as _machinery
import gc
import fitz  # PyMuPDF
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM
from openai import OpenAI
import logging

# Local tool imports
from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.utils.consts import MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.format_transformer import layoutjson2md
from qwen_vl_utils import process_vision_info


# --- Harmless flash_attn shim with a valid __spec__ --------------------------
def setup_flash_attn_shim():
    if "flash_attn" in sys.modules:
        return
    _m = types.ModuleType("flash_attn")
    _m.__spec__ = _machinery.ModuleSpec(name="flash_attn", loader=None)
    _m.__path__ = []
    def flash_attn_varlen_func(*args, **kwargs):
        raise RuntimeError("flash_attn is not available. Use attn_implementation='sdpa'.")
    _m.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["flash_attn"] = _m

# --- Model & System Setup ----------------------------------------------------
setup_flash_attn_shim()
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL_DIR = "./weights/DotsOCR"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32
MIN_PIX_MODEL = 256 * 28 * 28 * 4
MAX_PIX_MODEL = 640 * 28 * 28 * 4
DPI=200

# --- Global Model and Processor ---
MODEL = None
PROCESSOR = None


# --- M1/MPS-specific Helpers from m1.py -------------------------------------
def pil_from_page(page, dpi=DPI):
    pix = page.get_pixmap()
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def filter_json_blocks(obj, drop_headers_footers = True):
    """Removes header/footer blocks from the JSON object if requested."""
    if not drop_headers_footers or "blocks" not in obj:
        return obj
    obj["blocks"] = [
        b for b in obj.get("blocks", [])
        if b.get("category") not in ("Page-header", "Page-footer")
    ]
    return obj

def parse_json_flex(s: str):
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return json.loads(s)
    if s.startswith("[") and s.endswith("]"):
        return json.loads(s)
    m_obj = re.search(r'\{[\s\S]*\}', s)
    m_arr = re.search(r"[\s\S]*]", s)
    m = m_obj or m_arr
    if not m:
        raise ValueError("No JSON object or array found in the model output.")
    return json.loads(m.group(0))

def get_model_config(model_dir: str):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    if getattr(config, "vision_config", None) is not None:
        config.vision_config.attn_implementation = "sdpa"
    else:
        setattr(config, "attn_implementation", "sdpa")
    if hasattr(config, "use_sliding_window"):
        config.use_sliding_window = False
    if hasattr(config, "sliding_window"):
        config.sliding_window = None
    return config

def load_model_and_processor(model_dir: str, config, dtype, min_pix: int, max_pix: int):
    processor = AutoProcessor.from_pretrained(
        model_dir, local_files_only=True, min_pixels=min_pix,
        max_pixels=max_pix, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=True,
        config=config, attn_implementation="sdpa", torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    return model, processor

def patch_vision_tower(model):
    orig_forward_func = model.vision_tower.forward.__func__
    def _forward_no_bf16(self, hidden_states, grid_thw, bf16=True):
        return orig_forward_func(self, hidden_states, grid_thw, bf16=False)
    model.vision_tower.forward = types.MethodType(_forward_no_bf16, model.vision_tower)

def prepare_inputs(img, prompt_text, processor, device, dtype):
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            v = v.to(device).to(dtype=dtype) if torch.is_floating_point(v) else v.to(device)
            inputs[k] = v
    return inputs

def run_inference(model, inputs, max_new_tokens=15360):
    with torch.inference_mode():
        return model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=None, do_sample=False)

def decode_output(out_ids, inputs, processor):
    trimmed_ids = [o[len(iids):] for iids, o in zip(inputs["input_ids"], out_ids)]
    return processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

def extract_and_save_images(md_content, base_path, page_num):
    img_pattern = re.compile(r'!\[(.*?)\]\(data:image(?:\/(\w+))?;base64,([^)]+)\)')
    matches = list(img_pattern.finditer(md_content))
    
    for i, match in enumerate(matches):
        alt_text = match.group(1)
        img_type = match.group(2)
        b64_data = match.group(3)
        img_bytes = base64.b64decode(b64_data)
        
        if not img_type:
            try:
                with Image.open(io.BytesIO(img_bytes)) as img:
                    img_type = img.format.lower()
            except Exception:
                img_type = 'jpeg'

        img_filename = f"page_{page_num:03d}_image_{i+1}.{img_type}"
        img_path = os.path.join(base_path, img_filename)
        
        with open(img_path, "wb") as f:
            f.write(img_bytes)
            
        md_content = md_content.replace(match.group(0), f"![{alt_text}]({img_filename})")
        
    return md_content

def load_model():
    """Loads the model and processor."""
    global MODEL, PROCESSOR
    if MODEL is None or PROCESSOR is None:
        print(f"[INFO] Loading model from '{MODEL_DIR}' onto device '{DEVICE}'...")
        try:
            config = get_model_config(MODEL_DIR)
            MODEL, PROCESSOR = load_model_and_processor(MODEL_DIR, config, DTYPE, MIN_PIXELS, MAX_PIX_MODEL)
            MODEL.to(DEVICE)
            if DEVICE == "mps":
                patch_vision_tower(MODEL)
            print("[INFO] Model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load the model: {e}")
            MODEL = None
            PROCESSOR = None

def unload_model():
    """Unloads the model to free up memory."""
    global MODEL, PROCESSOR
    if MODEL is not None:
        print("[INFO] Unloading model to free up memory...")
        del MODEL
        del PROCESSOR
        MODEL = None
        PROCESSOR = None
        gc.collect()
        if DEVICE == "mps": torch.mps.empty_cache()
        print("[INFO] Model unloaded.")


# --- New Command-Line Specific Functions ---

def get_images_from_path(input_path: str) -> Tuple[List[Image.Image], List[Tuple[int, int]]]:
    """
    Loads images from a single file (PDF/image) or a directory of images.
    In directory mode, only image files are processed, not PDFs.
    """
    images = []
    original_sizes = []
    
    if os.path.isfile(input_path):
        file_ext = os.path.splitext(input_path)[1].lower()
        try:
            if file_ext == '.pdf':
                doc = fitz.open(input_path)
                for p in tqdm(doc, desc=f"Loading pages from {os.path.basename(input_path)}"):
                    images.append(pil_from_page(p, dpi=DPI))
                    original_sizes.append((p.rect.width, p.rect.height))
            elif file_ext in ['.jpg', '.jpeg', '.png']:
                img = Image.open(input_path)
                images.append(img)
                original_sizes.append((img.width, img.height))
            else:
                print(f"[WARNING] Unsupported file type: {file_ext}, skipping.")
        except Exception as e:
            print(f"[ERROR] Failed to load file {input_path}: {e}")
    
    elif os.path.isdir(input_path):
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(glob.glob(os.path.join(input_path, f"*{ext.lower()}")))
            image_files.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
        
        for file_path in tqdm(sorted(image_files), desc="Loading images from directory"):
            try:
                img = Image.open(file_path)
                images.append(img)
                original_sizes.append((img.width, img.height))
            except Exception as e:
                print(f"[ERROR] Failed to load image {file_path}: {e}")
    else:
        print(f"[ERROR] Input path is not a valid file or directory: {input_path}")

    return images, original_sizes

def process_page(img: Image.Image, page_num: int, prompt_text: str, output_dir: str, original_size: Tuple[int, int]):
    """
    Runs inference on a single page/image and saves the resulting artifacts.
    """
    out_text = None
    try:
        inputs = prepare_inputs(img, prompt_text, PROCESSOR, DEVICE, DTYPE)
        out_ids = run_inference(MODEL, inputs)
        out_text = decode_output(out_ids, inputs, PROCESSOR)
        
        parsed_json = parse_json_flex(out_text)
        
        if isinstance(parsed_json, list):
            page_info = {"blocks": parsed_json}
        else:
            page_info = parsed_json

        page_info = filter_json_blocks(page_info)

        original_width, original_height = original_size
        img_for_md = img.resize((int(original_width), int(original_height)))

        page_info.update({"page": page_num, "width": img_for_md.width, "height": img_for_md.height})
        
        md_content_base64 = layoutjson2md(img_for_md, page_info.get('blocks', []), text_key='text')
        md_content_paths = extract_and_save_images(md_content_base64, output_dir, page_num)

        # Save artifacts
        json_path = os.path.join(output_dir, f"page_{page_num:03d}.json")
        md_path = os.path.join(output_dir, f"page_{page_num:03d}.md")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(page_info, f, ensure_ascii=False, indent=2)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content_paths)

    except Exception as e:
        error_message = f"Failed to process page {page_num}: {e}, output: {out_text}"
        print(f"\n[ERROR] {error_message}")
        error_file_path = os.path.join(output_dir, f"page_{page_num:03d}_error.txt")
        with open(error_file_path, "w", encoding="utf-8") as f:
            f.write(error_message)
    finally:
        if 'inputs' in locals(): del inputs
        if 'out_ids' in locals(): del out_ids
        gc.collect()
        if DEVICE == "mps": torch.mps.empty_cache()

def main():
    """ Main function to run the batch processing. """

    parser = argparse.ArgumentParser(
        description="Batch OCR processing script using dots.ocr.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_path", type=str, help="Path to a single file (PDF/image) or a directory of images.")
    parser.add_argument("output_dir", type=str, help="Directory to save the output files.")
    parser.add_argument(
        "--prompt_mode", 
        type=str, 
        default="prompt_layout_all_en",
        help="The prompt mode to use. Available modes:\n" + "\n".join(dict_promptmode_to_prompt.keys())
    )

    args = parser.parse_args()

    # --- 1. Load Model ---
    load_model()
    if not MODEL or not PROCESSOR:
        print("[ERROR] Model failed to load. Check console for errors. Exiting.")
        sys.exit(1)

    # --- 2. Get Images ---
    images, original_sizes = get_images_from_path(args.input_path)
    if not images:
        print("[ERROR] No images found to process. Exiting.")
        unload_model()
        sys.exit(1)
    
    # --- 3. Setup Output ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Output will be saved to: {os.path.abspath(args.output_dir)}")

    # --- 4. Process Pages ---
    prompt_text = dict_promptmode_to_prompt.get(args.prompt_mode)
    if not prompt_text:
        print(f"[ERROR] Invalid prompt_mode: '{args.prompt_mode}'. Use --help to see available modes.")
        unload_model()
        sys.exit(1)

    print(f"[INFO] Starting processing for {len(images)} page(s)...")
    for i, img in enumerate(tqdm(images, desc="Processing pages")):
        process_page(img, i + 1, prompt_text, args.output_dir, original_sizes[i])

    # --- 5. Unload Model ---
    unload_model()
    print(f"\n[INFO] ✅ Batch processing complete. Results are in {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
