# dots_mps_parse.py
# macOS/M1-friendly runner for dots.ocr (local, offline, SDPA on MPS)
# - Uses repo's built-in prompts via --prompt-mode (JSON-ready)
# - Harmless flash_attn shim (with valid __spec__) to avoid import crashes
# - Forces SDPA attention (not Flash-Attn)
# - Disables Sliding-Window Attention for SDPA/Qwen2
# - Avoids BF16 by forcing FP16 everywhere and monkey-patching the vision tower
# - Processes PDFs page-by-page with pixel caps to avoid OOM on 16 GB
# - Parses JSON output flexibly (array OR object) and can drop headers/footers

import sys, types, importlib.machinery as _machinery
import os, gc, argparse, fitz, json, re
from PIL import Image
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from dots_ocr.utils.prompts import dict_promptmode_to_prompt  # repo prompts

# --- Harmless flash_attn shim with a valid __spec__ --------------------------
def setup_flash_attn_shim():
    """
    Creates a dummy 'flash_attn' module to prevent import errors on platforms
    where it's not available. The dummy function will raise a RuntimeError if called.
    """
    if "flash_attn" in sys.modules:
        return
    _m = types.ModuleType("flash_attn")
    _m.__spec__ = _machinery.ModuleSpec(name="flash_attn", loader=None)
    _m.__path__ = []
    def flash_attn_varlen_func(*args, **kwargs):
        raise RuntimeError(
            "flash_attn was requested but isn't available on this platform. "
            "Use attn_implementation='sdpa'."
        )
    _m.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["flash_attn"] = _m

# --- PDF & Image Utilities --------------------------------------------------
def pil_from_page(page, dpi=144):
    """Renders a PDF page to an RGB PIL.Image."""
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

# --- JSON Helpers -----------------------------------------------------------
def parse_json_flex(s: str):
    """
    Parses a string that may contain a JSON object or array, possibly
    surrounded by other text.
    """
    s = s.strip()
    # Fast paths for strings that are already valid JSON objects/arrays.
    if s.startswith("{") and s.endswith("}"):
        return json.loads(s)
    if s.startswith("[") and s.endswith("]"):
        return json.loads(s)

    # Fallback to regex for embedded JSON.
    m_obj = re.search(r"\{[\s\S]*\}", s)
    m_arr = re.search(r"\[[\s\S]*\]", s)
    
    if m_obj and m_arr:
        # If both are found, pick the one that appears first.
        m = m_obj if m_obj.start() < m_arr.start() else m_arr
    else:
        m = m_obj or m_arr
        
    if not m:
        raise ValueError("No JSON object or array found in the model output.")
    return json.loads(m.group(0))

def filter_json_blocks(obj, drop_headers_footers: bool):
    """Removes header/footer blocks from the JSON object if requested."""
    if not drop_headers_footers or "blocks" not in obj:
        return obj
    obj["blocks"] = [
        b for b in obj.get("blocks", [])
        if b.get("category") not in ("Page-header", "Page-footer")
    ]
    return obj

# --- Model Loading & Configuration ------------------------------------------
def get_model_config(model_dir: str):
    """Loads and configures the model settings for SDPA."""
    config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    # Force SDPA & disable Sliding-Window Attention (SWA)
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
    """Loads the model and processor from the specified directory."""
    processor = AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
        min_pixels=min_pix,
        max_pixels=max_pix,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        config=config,
        attn_implementation="sdpa",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    return model, processor

def patch_vision_tower(model):
    """
    Monkey-patches the vision tower to force FP16, avoiding dtype mismatches on MPS.
    """
    orig_forward_func = model.vision_tower.forward.__func__
    def _forward_no_bf16(self, hidden_states, grid_thw, bf16=True):
        return orig_forward_func(self, hidden_states, grid_thw, bf16=False)
    model.vision_tower.forward = types.MethodType(_forward_no_bf16, model.vision_tower)

# --- Inference & Processing -------------------------------------------------
def prepare_inputs(img, prompt_text, processor, device, dtype):
    """Prepares model inputs from an image and prompt text."""
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt_text},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")

    # Move tensors to the correct device and dtype
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if torch.is_floating_point(v):
                v = v.to(dtype=dtype)
            inputs[k] = v
    return inputs

def run_inference(model, inputs, max_new_tokens):
    """Runs the model to generate token IDs."""
    with torch.inference_mode():
        return model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=None,
            do_sample=False,
        )

def decode_output(out_ids, inputs, processor):
    """Decodes the generated token IDs to text."""
    trimmed_ids = [o[len(iids):] for iids, o in zip(inputs["input_ids"], out_ids)]
    return processor.batch_decode(
        trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

def save_result(obj, json_out_path):
    """Saves the JSON object to a .jsonl file."""
    out_line = json.dumps(obj, ensure_ascii=False)
    # Ensure the output path ends with .jsonl
    if not json_out_path.lower().endswith(".jsonl"):
        json_out_path += ".jsonl"
    with open(json_out_path, "a", encoding="utf-8") as f:
        f.write(out_line + "\n")

# --- Main Orchestration -----------------------------------------------------
def get_cli_args():
    """Parses and returns command-line arguments."""
    PROMPT_KEYS = tuple(dict_promptmode_to_prompt.keys())
    DEFAULT_PROMPT_MODE = "prompt_layout_all_en" if "prompt_layout_all_en" in PROMPT_KEYS else (PROMPT_KEYS[0] if PROMPT_KEYS else None)

    ap = argparse.ArgumentParser(description="macOS/M1-friendly runner for dots.ocr.")
    ap.add_argument("pdf", help="Path to the PDF file to process.")
    ap.add_argument("--model-dir", default="./DotsOCR", help="Directory of the DotsOCR model.")
    ap.add_argument("--dpi", type=int, default=300, help="DPI for rendering PDF pages.")
    ap.add_argument("--max-new-tokens", type=int, default=1536, help="Max new tokens for the model.")
    ap.add_argument("--prompt", default=None, help="Custom user prompt (overrides --prompt-mode).")
    ap.add_argument("--prompt-mode",
                    default=DEFAULT_PROMPT_MODE,
                    choices=list(PROMPT_KEYS),
                    help="Use a built-in dots.ocr prompt.")
    ap.add_argument("--json-out", default=None,
                    help="Path to write JSONL output. Appends one JSON object per page.")
    ap.add_argument("--drop-headers-footers", action="store_true",
                    help="Drop Page-header/Page-footer blocks from JSON.")
    ap.add_argument("--print-raw", action="store_true",
                    help="Print raw model output before JSON parsing.")
    return ap.parse_args()

def main():
    """Main function to set up and run the OCR process."""
    args = get_cli_args()
    setup_flash_attn_shim()

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    # Cap visual tokens to avoid OOM on 16 GB
    min_pix = 256 * 28 * 28 * 4
    max_pix = 640 * 28 * 28 * 4

    config = get_model_config(args.model_dir)
    model, processor = load_model_and_processor(args.model_dir, config, dtype, min_pix, max_pix)
    
    model.to(device)
    if device == "mps":
        patch_vision_tower(model)

    attn_impl = getattr(getattr(model.config, "vision_config", model.config), "attn_implementation", None)
    print(f"[info] device={device}, dtype={dtype}, vision_attn={attn_impl}, "
          f"use_sliding_window={getattr(model.config, 'use_sliding_window', None)}")

    doc = fitz.open(args.pdf)
    for i, page in enumerate(doc, start=1):
        img = pil_from_page(page, dpi=args.dpi)

        prompt_text = dict_promptmode_to_prompt.get(args.prompt_mode, "")
        if args.prompt is not None:
            prompt_text = args.prompt

        inputs = prepare_inputs(img, prompt_text, processor, device, dtype)
        out_ids = run_inference(model, inputs, args.max_new_tokens)
        out_text = decode_output(out_ids, inputs, processor)

        if args.print_raw:
            print("\n--- RAW MODEL OUTPUT ---\n")
            print(out_text)

        try:
            parsed = parse_json_flex(out_text)
            obj = {"blocks": parsed} if isinstance(parsed, list) else parsed

            w, h = img.size
            obj.setdefault("page", i)
            obj.setdefault("width", w)
            obj.setdefault("height", h)

            obj = filter_json_blocks(obj, args.drop_headers_footers)

            print(f"\n===== Page {i}/{len(doc)} JSON =====\n")
            print(json.dumps(obj, ensure_ascii=False, indent=2))

            if args.json_out:
                save_result(obj, args.json_out)

        except ValueError as e:
            print(f"\n--- ERROR: Could not parse JSON for page {i} ---")
            print(f"Error: {e}")
            print("Raw output was:")
            print(out_text)
            print("-------------------------------------------------")


        # Free per-page memory
        del img, inputs, out_ids, out_text
        if 'obj' in locals(): del obj
        if 'parsed' in locals(): del parsed
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

if __name__ == "__main__":
    main()