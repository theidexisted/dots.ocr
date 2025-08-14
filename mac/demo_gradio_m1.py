"""
Gradio Demo for dots.ocr on macOS/M1
- Runs the model locally using MPS acceleration.
- Self-contained, no external server needed.
- Based on the original Gradio demo and the M1 command-line script.
"""

import gradio as gr
import json
import os
import io
import tempfile
import base64
import zipfile
import uuid
import re
from pathlib import Path
from PIL import Image
import shutil
import sys
import types
import importlib.machinery as _machinery
import gc
import fitz  # PyMuPDF
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM

# Local tool imports
from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.utils.consts import MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.doc_utils import load_images_from_pdf
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
# Cap visual tokens to avoid OOM on 16 GB
MIN_PIX_MODEL = 256 * 28 * 28 * 4
MAX_PIX_MODEL = 640 * 28 * 28 * 4

# --- Global Model and Processor ---
# These are loaded once at startup.
MODEL = None
PROCESSOR = None

# --- M1/MPS-specific Helpers from m1.py -------------------------------------
def pil_from_page(page, dpi=200):
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def parse_json_flex(s: str):
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return json.loads(s)
    if s.startswith("[") and s.endswith("]"):
        return json.loads(s)
    m_obj = re.search(r"\{[\s\S]*\}", s)
    m_arr = re.search(r"\[[\s\S]*\]", s)
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

def run_inference(model, inputs, max_new_tokens=1536):
    with torch.inference_mode():
        return model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=None, do_sample=False)

def decode_output(out_ids, inputs, processor):
    trimmed_ids = [o[len(iids):] for iids, o in zip(inputs["input_ids"], out_ids)]
    return processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# --- Gradio App Helpers (Adapted from demo_gradio.py) -----------------------
def get_initial_session_state():
    return {
        'processing_results': { 'temp_dir': None, 'session_id': None },
        'pdf_cache': {
            "images": [], "current_page": 0, "total_pages": 0,
            "file_type": None, "is_parsed": False, "results": []
        }
    }

def create_temp_session_dir():
    session_id = uuid.uuid4().hex[:8]
    temp_dir = os.path.join(tempfile.gettempdir(), f"dots_ocr_demo_{session_id}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir, session_id

def load_file_for_preview(file_path, session_state):
    pdf_cache = session_state['pdf_cache']
    if not file_path or not os.path.exists(file_path):
        return None, "<div id='page_info_box'>0 / 0</div>", session_state
    
    file_ext = os.path.splitext(file_path)[1].lower()
    try:
        if file_ext == '.pdf':
            pages = [pil_from_page(p, dpi=200) for p in fitz.open(file_path)]
            pdf_cache["file_type"] = "pdf"
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            pages = [Image.open(file_path)]
            pdf_cache["file_type"] = "image"
        else:
            return None, "<div id='page_info_box'>Unsupported</div>", session_state
    except Exception as e:
        return None, f"<div id='page_info_box'>Load failed: {e}</div>", session_state
    
    pdf_cache.update({"images": pages, "current_page": 0, "total_pages": len(pages), "is_parsed": False, "results": []})
    return pages[0], f"<div id='page_info_box'>1 / {len(pages)}</div>", session_state

def turn_page(direction, session_state):
    pdf_cache = session_state['pdf_cache']
    if not pdf_cache["images"]:
        return None, "<div id='page_info_box'>0 / 0</div>", "", session_state

    if direction == "prev":
        pdf_cache["current_page"] = max(0, pdf_cache["current_page"] - 1)
    else: # "next"
        pdf_cache["current_page"] = min(pdf_cache["total_pages"] - 1, pdf_cache["current_page"] + 1)

    index = pdf_cache["current_page"]
    current_image = pdf_cache["images"][index]
    page_info = f"<div id='page_info_box'>{index + 1} / {pdf_cache['total_pages']}</div>"
    
    current_json_str = ""
    if pdf_cache["is_parsed"] and index < len(pdf_cache["results"]):
        result = pdf_cache["results"][index]
        current_json_str = json.dumps(result.get('json_data', {}), ensure_ascii=False, indent=2)
    
    return current_image, page_info, current_json_str, session_state

def clear_all_data(session_state):
    temp_dir = session_state['processing_results'].get('temp_dir')
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    new_session_state = get_initial_session_state()
    return (
        None, "", None, "Waiting for processing...", "## Waiting for processing...",
        "...", gr.update(visible=False), "<div id='page_info_box'>0 / 0</div>",
        "...", new_session_state
    )

def update_prompt_display(prompt_mode):
    return dict_promptmode_to_prompt.get(prompt_mode, "Unknown prompt mode.")

# --- Core Local Inference Function ------------------------------------------
def process_file_local(session_state, test_image_input, file_input, prompt_mode):
    if not MODEL or not PROCESSOR:
        return None, "Error: Model not loaded. Check console for errors.", "", "", gr.update(value=None), None, "", session_state

    input_path = file_input if file_input else test_image_input
    if not input_path:
        return None, "Please upload a file or select an example.", "", "", gr.update(value=None), None, "", session_state

    # Setup temp dir
    temp_dir, session_id = create_temp_session_dir()
    session_state['processing_results'].update({'temp_dir': temp_dir, 'session_id': session_id})

    pdf_cache = session_state['pdf_cache']
    images_to_process = pdf_cache.get("images", [])
    if not images_to_process: # If file wasn't previewed, load it now
        _, _, session_state = load_file_for_preview(input_path, session_state)
        images_to_process = session_state['pdf_cache'].get("images", [])

    prompt_text = dict_promptmode_to_prompt.get(prompt_mode, "")
    all_results = []
    all_md_content = []

    for i, img in enumerate(images_to_process):
        page_num = i + 1
        try:
            inputs = prepare_inputs(img, prompt_text, PROCESSOR, DEVICE, DTYPE)
            out_ids = run_inference(MODEL, inputs)
            out_text = decode_output(out_ids, inputs, PROCESSOR)
            
            parsed_json = parse_json_flex(out_text)
            page_info = {"blocks": parsed_json} if isinstance(parsed_json, list) else parsed_json
            page_info.update({"page": page_num, "width": img.width, "height": img.height})
            
            md_content = layoutjson2md(img, page_info.get('blocks', []), text_key='text')
            
            all_results.append({'json_data': page_info, 'md_content': md_content})
            all_md_content.append(md_content)

            # Save artifacts
            json_path = os.path.join(temp_dir, f"page_{page_num}.json")
            md_path = os.path.join(temp_dir, f"page_{page_num}.md")
            with open(json_path, "w", encoding="utf-8") as f: json.dump(page_info, f, ensure_ascii=False, indent=2)
            with open(md_path, "w", encoding="utf-8") as f: f.write(md_content)

        except Exception as e:
            error_message = f"Failed to process page {page_num}: {e}"
            all_results.append({'json_data': {'error': error_message}, 'md_content': f"## {error_message}"})
            all_md_content.append(f"## {error_message}")
        finally:
            # Clean up memory after each page
            del img
            if 'inputs' in locals(): del inputs
            if 'out_ids' in locals(): del out_ids
            gc.collect()
            if DEVICE == "mps": torch.mps.empty_cache()

    pdf_cache.update({"is_parsed": True, "results": all_results})
    
    # Prepare final outputs for UI
    first_result = all_results[0]
    first_image = images_to_process[0]
    combined_md = "\n\n---\n\n".join(all_md_content)
    first_page_json_str = json.dumps(first_result.get('json_data', {}), ensure_ascii=False, indent=2)
    
    info_text = f"**Local Inference Details:**\n- Device: {DEVICE}\n- Total Pages: {len(images_to_process)}\n- Session ID: {session_id}"
    
    # Create zip for download
    download_zip_path = os.path.join(temp_dir, f"results_{session_id}.zip")
    with zipfile.ZipFile(download_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if not file.endswith('.zip'):
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), temp_dir))

    return (
        first_image, info_text, combined_md, combined_md,
        gr.update(value=download_zip_path, visible=True),
        f"<div id='page_info_box'>1 / {len(images_to_process)}</div>",
        first_page_json_str, session_state
    )

# --- Gradio Interface Creation ----------------------------------------------
def create_gradio_interface():
    css = """
    #parse_button { background: #FF576D !important; border-color: #FF576D !important; }
    #parse_button:hover { background: #F72C49 !important; border-color: #F72C49 !important; }
    #page_info_html { display: flex; align-items: center; justify-content: center; height: 100%; margin: 0 12px; }
    #page_info_box { padding: 8px 20px; font-size: 16px; border: 1px solid #bbb; border-radius: 8px; background-color: #f8f8f8; text-align: center; min-width: 80px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    #markdown_output { min-height: 800px; overflow: auto; }
    footer { visibility: hidden; }
    #info_box { padding: 10px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6; margin: 10px 0; font-size: 14px; }
    """
    with gr.Blocks(theme="ocean", css=css, title='dots.ocr (M1 Local)') as demo:
        session_state = gr.State(value=get_initial_session_state())
        
        gr.HTML("""
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
                <h1 style="margin: 0; font-size: 2em;">🔍 dots.ocr (macOS/M1 Local)</h1>
            </div>
            <div style="text-align: center; margin-bottom: 10px;">
                <em>Local layout analysis on Apple Silicon using MPS</em>
            </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Upload & Select")
                file_input = gr.File(label="Upload PDF/Image", type="filepath", file_types=[".pdf", ".jpg", ".jpeg", ".png"])
                
                test_images_dir = "./assets/showcase_origin"
                test_images = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                test_image_input = gr.Dropdown(label="Or Select an Example", choices=[""] + test_images, value="")

                gr.Markdown("### ⚙️ Prompt & Actions")
                prompt_mode = gr.Dropdown(label="Select Prompt", choices=list(dict_promptmode_to_prompt.keys()), value="prompt_layout_all_en")
                prompt_display = gr.Textbox(label="Current Prompt Content", value=dict_promptmode_to_prompt["prompt_layout_all_en"], lines=4, max_lines=8, interactive=False, show_copy_button=True)
                
                with gr.Row():
                    process_btn = gr.Button("🔍 Parse Locally", variant="primary", scale=2, elem_id="parse_button")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
            
            with gr.Column(scale=6):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### 👁️ File Preview")
                        result_image = gr.Image(label="Layout Preview", height=800, show_label=False)
                        with gr.Row():
                            prev_btn = gr.Button("⬅ Previous", size="sm")
                            page_info = gr.HTML(value="<div id='page_info_box'>0 / 0</div>", elem_id="page_info_html")
                            next_btn = gr.Button("Next ➡", size="sm")
                        info_display = gr.Markdown("Waiting for processing...", elem_id="info_box")
                    
                    with gr.Column(scale=3):
                        gr.Markdown("### ✔️ Result Display")
                        with gr.Tabs():
                            with gr.TabItem("Markdown Render"):
                                md_output = gr.Markdown("## Click 'Parse Locally' to begin...", elem_id="markdown_output", latex_delimiters=[{"left": "$$", "right": "$$", "display": True}])
                            with gr.TabItem("Markdown Raw"):
                                md_raw_output = gr.Textbox("...", label="Markdown Raw Text", lines=38, show_copy_button=True, show_label=False)
                            with gr.TabItem("Current Page JSON"):
                                current_page_json = gr.Textbox("...", label="Current Page JSON", lines=38, show_copy_button=True, show_label=False)
                
                download_btn = gr.DownloadButton("⬇️ Download Results", visible=False)

        # Event Handlers
        prompt_mode.change(fn=update_prompt_display, inputs=prompt_mode, outputs=prompt_display)
        file_input.upload(fn=load_file_for_preview, inputs=[file_input, session_state], outputs=[result_image, page_info, session_state])
        test_image_input.change(fn=load_file_for_preview, inputs=[test_image_input, session_state], outputs=[result_image, page_info, session_state])
        
        prev_btn.click(fn=lambda s: turn_page("prev", s), inputs=[session_state], outputs=[result_image, page_info, current_page_json, session_state])
        next_btn.click(fn=lambda s: turn_page("next", s), inputs=[session_state], outputs=[result_image, page_info, current_page_json, session_state])
        
        process_btn.click(
            fn=process_file_local,
            inputs=[session_state, test_image_input, file_input, prompt_mode],
            outputs=[result_image, info_display, md_output, md_raw_output, download_btn, page_info, current_page_json, session_state]
        )
        clear_btn.click(
            fn=clear_all_data, inputs=[session_state],
            outputs=[file_input, test_image_input, result_image, info_display, md_output, md_raw_output, download_btn, page_info, current_page_json, session_state]
        )
    return demo

# --- Main Program Execution -------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Starting local M1 Gradio demo...")
    if not os.path.exists(MODEL_DIR):
        print(f"[ERROR] Model directory not found at '{MODEL_DIR}'. Please download the model first.")
        sys.exit(1)

    print(f"[INFO] Loading model from '{MODEL_DIR}' onto device '{DEVICE}'...")
    try:
        config = get_model_config(MODEL_DIR)
        MODEL, PROCESSOR = load_model_and_processor(MODEL_DIR, config, DTYPE, MIN_PIX_MODEL, MAX_PIX_MODEL)
        MODEL.to(DEVICE)
        if DEVICE == "mps":
            patch_vision_tower(MODEL)
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load the model: {e}")
        sys.exit(1)

    demo = create_gradio_interface()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    print(f"[INFO] Launching Gradio interface on http://0.0.0.0:{port}")
    demo.queue().launch(server_name="0.0.0.0", server_port=port, debug=True)
