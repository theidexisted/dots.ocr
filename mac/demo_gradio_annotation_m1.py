"""
Gradio Web Application for dots.ocr on macOS/M1 with Annotation
"""

import gradio as gr
import json
import os
import io
import tempfile
import base64
import uuid
import re
from pathlib import Path
from PIL import Image
import shutil
import sys
import types
import importlib.machinery as _machinery
import gc

import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from dots_ocr.utils.prompts import dict_promptmode_to_prompt
from dots_ocr.utils.format_transformer import layoutjson2md
from gradio_image_annotation import image_annotator

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

    def run(self, image, prompt_mode, bbox=None):
        # For M1, bbox is handled by modifying the prompt directly
        prompt_text = dict_promptmode_to_prompt.get(prompt_mode, "")
        if bbox:
            # Assuming prompt_grounding_ocr expects bbox as part of the prompt
            # This is a simplification; actual implementation might need image resizing/bbox scaling
            prompt_text = prompt_text + str(bbox) # Append bbox to prompt for grounding

        inputs = self.prepare_inputs(image, prompt_text)
        out_ids = self.run_inference(inputs)
        out_text = self.decode_output(out_ids, inputs)
        
        del inputs, out_ids
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
            
        return out_text

# ==================== Global Variables ====================
dots_ocr_m1 = DotsOCRM1()
DEFAULT_CONFIG = {
    'test_images_dir': "./assets/showcase_origin",
}

# ==================== Utility Functions ====================
def read_image_v2(img_path):
    """Reads an image from a path, supporting PIL Image objects as well."""
    if isinstance(img_path, Image.Image):
        return img_path
    elif isinstance(img_path, str):
        return Image.open(img_path).convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(img_path)}")

def get_test_images():
    """Gets the list of test images."""
    test_images = []
    test_dir = DEFAULT_CONFIG['test_images_dir']
    if os.path.exists(test_dir):
        test_images = [os.path.join(test_dir, name) for name in os.listdir(test_dir) 
                      if name.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return test_images

def process_annotation_data(annotation_data):
    """Processes annotation data, converting it to the format required by the model."""
    if not annotation_data or not annotation_data.get('boxes'):
        return None, None
    
    image = annotation_data.get('image')
    boxes = annotation_data.get('boxes', [])
    
    if not boxes:
        return image, None
    
    # Ensure the image is in PIL Image format
    if image is not None:
        import numpy as np
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            try:
                image = Image.open(image) if isinstance(image, str) else Image.fromarray(image)
            except Exception as e:
                print(f"Image format conversion failed: {e}")
                return None, None
    
    # Get the coordinate information of the box (only one box) and convert to int
    box = boxes[0]
    bbox = [int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])]
    
    return image, bbox

# ==================== Core Processing Function ====================
def process_image_inference_with_annotation(annotation_data, test_image_input, prompt_mode):
    """Core function for image inference, supporting annotation data."""
    
    image = None
    bbox = None
    raw_json_output = ""
    md_content = ""
    info_text = ""

    # Prioritize processing annotation data
    if annotation_data and annotation_data.get('image') is not None:
        image, bbox = process_annotation_data(annotation_data)
        if image is not None and bbox is None: # Image provided but no box drawn
            return "Please draw a bounding box on the image.", "", "", gr.update(visible=False), ""
    
    # If no annotation data, check the test image input
    if image is None and test_image_input and test_image_input != "":
        try:
            image = read_image_v2(test_image_input)
        except Exception as e:
            return f"Failed to read test image: {e}", "", "", gr.update(visible=False), ""
    
    if image is None:
        return "Please select a test image or add an image in the annotation component", "", "", gr.update(visible=False), ""
    
    try:
        # Run inference using the M1-optimized model
        raw_output = dots_ocr_m1.run(image, prompt_mode, bbox=bbox)
        
        # Parse the raw output (expected to be JSON)
        parsed_json = dots_ocr_m1.parse_json_flex(raw_output)
        raw_json_output = json.dumps(parsed_json, ensure_ascii=False, indent=2)
        
        # Convert JSON to Markdown
        md_content = layoutjson2md(image, parsed_json, text_key='text')
        
        info_text = f"""
**Image Information:**
- Original Dimensions: {image.width} x {image.height}
- Processing Mode: {'Region OCR' if bbox else 'Full Image OCR'}
- Box Coordinates: {bbox if bbox else 'None'}
        """
        
        return md_content, info_text, md_content, gr.update(value=raw_json_output, visible=True), raw_json_output
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"An error occurred during processing: {e}", "", "", gr.update(visible=False), ""

def load_image_to_annotator(test_image_input):
    """Loads an image into the annotation component."""
    image = None
    if test_image_input and test_image_input != "":
        try:
            image = read_image_v2(test_image_input)
        except Exception as e:
            return None
    
    if image is None:
        return None
    
    return {"image": image, "boxes": []}

def clear_all_data():
    """Clears all data."""
    return (
        "",    # Clear test image selection
        None,  # Clear annotation component
        "Waiting for processing results...",  # Reset info display
        "## Waiting for processing results...",  # Reset Markdown display
        "🕐 Waiting for parsing results...",    # Clear raw Markdown text
        gr.update(visible=False),  # Hide download button
        "🕐 Waiting for parsing results..."     # Clear JSON
    )

def update_prompt_display(prompt_mode):
    """Updates the displayed prompt content."""
    return dict_promptmode_to_prompt[prompt_mode]

# ==================== Gradio Interface ====================
def create_gradio_interface():
    """Creates the Gradio interface."""
    
    css = """
    footer { visibility: hidden; }
    #info_box { padding: 10px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6; margin: 10px 0; font-size: 14px; }
    #markdown_tabs { height: 100%; }
    #annotation_component { border-radius: 8px; }
    """
    
    with gr.Blocks(theme="ocean", css=css, title='dots.ocr - Annotation (M1)') as demo:
        
        gr.HTML("""
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
                <h1 style="margin: 0; font-size: 2em;">🔍 dots.ocr - Annotation Version (macOS/M1)</h1>
            </div>
            <div style="text-align: center; margin-bottom: 10px;">
                <em>Supports image annotation, drawing boxes, and sending box information to the model for OCR.</em>
            </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1, variant="compact"):
                gr.Markdown("### 📁 Select Example")
                test_images = get_test_images()
                test_image_input = gr.Dropdown(
                    label="Select Example",
                    choices=[""] + test_images,
                    value="",
                    show_label=True
                )
                
                load_btn = gr.Button("📷 Load Image to Annotation Area", variant="secondary")
                
                prompt_mode = gr.Dropdown(
                    label="Select Prompt",
                    choices=["prompt_grounding_ocr"], # Only grounding OCR makes sense with annotation
                    value="prompt_grounding_ocr",
                    show_label=True,
                    info="'prompt_grounding_ocr' mode will be used automatically with a drawn box."
                )
                
                prompt_display = gr.Textbox(
                    label="Current Prompt Content",
                    value=dict_promptmode_to_prompt["prompt_grounding_ocr"],
                    lines=4,
                    max_lines=8,
                    interactive=False,
                    show_copy_button=True
                )
                
                gr.Markdown("### ⚙️ Actions")
                process_btn = gr.Button("🔍 Parse", variant="primary")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                
            with gr.Column(scale=6, variant="compact"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### 🎯 Image Annotation Area")
                        gr.Markdown("""
                        **Instructions:**
                        - Method 1: Select an example image on the left and click "Load Image to Annotation Area".
                        - Method 2: Upload an image directly in the annotation area below (drag and drop or click to upload).
                        - Use the mouse to draw a box on the image to select the region for recognition.
                        - Only one box can be drawn. To draw a new one, please delete the old one first.
                        - **Hotkey: Press the Delete key to remove the selected box.**
                        - After drawing a box, clicking Parse will automatically use the Region OCR mode.
                        """)
                        
                        annotator = image_annotator(
                            value=None,
                            label="Image Annotation",
                            height=600,
                            show_label=False,
                            elem_id="annotation_component",
                            single_box=True,
                            box_min_size=10,
                            interactive=True,
                            disable_edit_boxes=True,
                            label_list=["OCR Region"],
                            label_colors=[(255, 0, 0)],
                            use_default_label=True,
                            image_type="pil"
                        )
                        
                        info_display = gr.Markdown(
                            "Waiting for processing results...",
                            elem_id="info_box"
                        )
                    
                    with gr.Column(scale=3):
                        gr.Markdown("### ✅ Results")
                        
                        with gr.Tabs(elem_id="markdown_tabs"):
                            with gr.TabItem("Markdown Rendered View"):
                                md_output = gr.Markdown(
                                    "## Please upload an image and click the Parse button for recognition...",
                                    label="Markdown Preview",
                                    max_height=1000,
                                    latex_delimiters=[
                                        {"left": "$$", "right": "$$", "display": True},
                                        {"left": "$", "right": "$", "display": False},
                                    ],
                                    show_copy_button=False,
                                    elem_id="markdown_output"
                                )
                            
                            with gr.TabItem("Markdown Raw Text"):
                                md_raw_output = gr.Textbox(
                                    value="🕐 Waiting for parsing results...",
                                    label="Markdown Raw Text",
                                    max_lines=100,
                                    lines=38,
                                    show_copy_button=True,
                                    elem_id="markdown_output",
                                    show_label=False
                                )
                            
                            with gr.TabItem("JSON Result"):
                                json_output = gr.Textbox(
                                    value="🕐 Waiting for parsing results...",
                                    label="JSON Result",
                                    max_lines=100,
                                    lines=38,
                                    show_copy_button=True,
                                    elem_id="markdown_output",
                                    show_label=False
                                )
                
                with gr.Row():
                    download_btn = gr.DownloadButton(
                        "⬇️ Download JSON Result",
                        visible=False
                    )
        
        prompt_mode.change(
            fn=update_prompt_display,
            inputs=prompt_mode,
            outputs=prompt_display,
            show_progress=False
        )
        
        load_btn.click(
            fn=load_image_to_annotator,
            inputs=[test_image_input],
            outputs=annotator,
            show_progress=False
        )
        
        process_btn.click(
            fn=process_image_inference_with_annotation,
            inputs=[
                annotator, test_image_input, prompt_mode
            ],
            outputs=[
                md_output, info_display, md_raw_output, download_btn, json_output
            ],
            show_progress=True
        )
        
        clear_btn.click(
            fn=clear_all_data,
            outputs=[
                test_image_input, annotator,
                info_display, md_output, md_raw_output,
                download_btn, json_output
            ],
            show_progress=False
        )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.queue().launch(
        server_name="0.0.0.0", 
        server_port=7862,  # Use a different port to avoid conflicts
        debug=True
    )
