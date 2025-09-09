'''
Gradio Demo for dots.ocr on macOS/M1
- Runs the model locally using MPS acceleration.
- Self-contained, no external server needed.
- Based on the original Gradio demo and the M1 command-line script.
'''
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
import uvicorn
from fastapi import FastAPI, HTTPException
from starlette.responses import RedirectResponse
from openai import OpenAI
import logging
from tqdm import tqdm
import glob
from typing import List, Tuple
from contextlib import asynccontextmanager

pageLimit = 0

class PDFMerger:
    def __init__(self, api_base: str, model: str = "gpt-3.5-turbo"):
        """初始化PDF合并器"""
        self.api_base = api_base
        self.model = model
        
        # 创建独立的logger实例
        self.logger = logging.getLogger("PDFMerger")
        self.logger.setLevel(logging.INFO)
        
        # 避免重复添加handler
        if not self.logger.handlers:
            # 文件handler
            file_handler = logging.FileHandler("merger.log", encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # 控制台handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 格式化
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            #console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            #self.logger.addHandler(console_handler)
        
        # 配置OpenAI客户端
        self.client = OpenAI(
            base_url=api_base,
            api_key="dummy" # The key can be anything, it is ignored by the local server
        )

    def read_ocr_files(self, input_dir: str) -> List[Tuple[str, str]]:
        """读取所有OCR文件内容和路径"""
        file_pattern = os.path.join(input_dir, "*.md")
        files = sorted(glob.glob(file_pattern), key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))

        global pageLimit
        if pageLimit is not None and pageLimit > 0:
            files = files[:pageLimit]
            self.logger.info(f"限制处理前 {pageLimit} 页")
        
        contents = []
        for file_path in tqdm(files, desc="读取OCR文件"):
            with open(file_path, 'r', encoding='utf-8') as f:
                contents.append((file_path, f.read().strip()))
        
        return contents
    def remove_think_blocks_simple(self, text):
        """
        Remove <think>...</think> blocks using string methods.
        
        Args:
            text (str): Input string containing <think> blocks
            
        Returns:
            str: String with all <think> blocks removed
        """
        result = text
        while True:
            start = result.find('<think>')
            if start == -1:
                break
                
            end = result.find('</think>', start)
            if end == -1:
                break
                
            # Remove the block including the tags
            result = result[:start] + result[end + 8:]  # 8 = len('</think>')
        
        return result

    def remove_unnecessary_spaces(self, content: str, file_path: str) -> str:
        """使用AI删除正文中不应该存在的空格"""
        prompt = f"""
请处理以下 markdown 文本，删除正文中不应该存在的空格，但保留标题、列表等特殊格式中的空格。同时处理一下图片说明文字，把图片下面的说文字移动到 markdown 的 caption 位置

处理规则：
1. 删除中文文本中不必要的空格（中文通常不需要单词间的空格）
2. 保留英文单词之间的必要空格
3. 保留标题、列表项等格式中的空格
4. 保留代码块、URL等特殊内容中的空格
5. 保留标点符号周围的正常格式
6. 修复标题换行问题，确保标题都是单独一行，且标题前后有空行
7. 同时注意标题是否中间混入了 # 字符，如果有请删掉


对于图片说明的处理示例：

```
![](page_001_image_1.png)

图2-1局部反映整体
```

修改为：

```
![图2-1局部反映整体](page_001_image_1.png)
```



请直接返回处理后的文本，不要添加任何解释。

需要处理的文本：
{content}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个 Markdown 文本处理助手，专门清理文本中不必要的空格。以及修复标题换行问题"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1  # 低温度以确保一致性
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"空格清理API调用失败: {os.path.basename(file_path)}")
            self.logger.error(f"错误: {e}")
            # 如果API调用失败，返回原始内容
            return content

    def preprocess_ocr_contents(self, contents_with_paths: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """预处理OCR内容，删除不必要的空格"""
        processed_contents = []
        
        for file_path, content in tqdm(contents_with_paths, desc="清理空格"):
            if content.strip():  # 只处理非空内容
                processed_content = self.remove_think_blocks_simple(self.remove_unnecessary_spaces(content, file_path))
                self.logger.info(f"已处理空格: {os.path.basename(file_path)}, old content:\n{content}\nnew content:\n{processed_content}")
                processed_contents.append((file_path, processed_content))
            else:
                processed_contents.append((file_path, content))
        
        return processed_contents

    def should_merge_pages(self, last_line: str, next_first_line: str, prev_path: str, current_path: str) -> bool:
        """使用AI判断两个页面之间的文本是否应该连接"""
        prompt = f"""
这两行文字是一本书里面连续两页分别做 OCR 的结果，现在根据上下文判断它们是否属于同一段落。
上一页的最后一行是: "{last_line}"
下一页的第一行是: "{next_first_line}"

如果是同一段落请返回 True，否则返回 False。 请不要返回除此以外的其它内容。

主要的思路：

* 如果上一行没有以句号、问号、感叹号、引号结尾，则很可能是同一段落。
* 如果下一行以中文标点开头，则很可能是同一段落。
* 如果其中一个是标题，那肯定不是同一段落。
* 其它情况要结合上下文进行判断。

"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个文本分析助手，专门判断段落连续性。"},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content.strip().lower()
            return "true" in answer
        except Exception as e:
            request_data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是一个文本分析助手，专门判断段落连续性。"},
                    {"role": "user", "content": prompt}
                ]
            }
            self.logger.error(f"API调用失败，处理文件: {os.path.basename(prev_path)} 和 {os.path.basename(current_path)}")
            self.logger.error(f"错误: {e}")
            self.logger.error(f"请求JSON:\n{json.dumps(request_data, indent=2, ensure_ascii=False)}")
            # 如果API调用失败，默认为不合并
            return False

    def extract_context(self, page_content: str) -> Tuple[str, str]:
        """提取页面的首行和末行"""
        lines = [line.strip() for line in page_content.split("\n") if line.strip()]
        first_line = lines[0] if lines else ""
        last_line = lines[-1] if lines else ""
        return first_line, last_line

    def merge_ocr_contents(self, contents_with_paths: List[Tuple[str, str]]) -> str:
        """合并OCR内容，处理页面间的连接问题"""
        if not contents_with_paths:
            return ""

        paths = [p for p, c in contents_with_paths]
        contents = [c for p, c in contents_with_paths]

        merged_text = contents[0]
        
        for i in tqdm(range(1, len(contents)), desc="合并页面"):
            prev_page_content = contents[i-1]
            current_page_content = contents[i]
            
            prev_page_path = paths[i-1]
            current_page_path = paths[i]

            _, prev_last_line = self.extract_context(prev_page_content)
            current_first_line, _ = self.extract_context(current_page_content)
            self.logger.info(f"处理页面: {os.path.basename(prev_page_path)} 和 {os.path.basename(current_page_path)}")
            self.logger.info(f"上一页最后一行: {prev_last_line}")
            self.logger.info(f"下一页第一行: {current_first_line}")

            if not prev_last_line or not current_first_line:
                self.logger.info("其中一页为空，直接添加换行")
                merged_text += "\n\n" + current_page_content
                continue

            if self.should_merge_pages(prev_last_line, current_first_line, prev_page_path, current_page_path):
                self.logger.info("判断为同一段落，直接连接")
                lines = current_page_content.split('\n')
                merged_text = merged_text.rstrip() + lines[0]
                if len(lines) > 1:
                    merged_text += "\n" + "\n".join(lines[1:])
            else:
                self.logger.info("判断不是同一段落，添加换行")
                merged_text += "\n\n" + current_page_content
        
        return merged_text

    def process(self, input_dir: str, output_file: str) -> None:
        """处理OCR文件并生成合并后的Markdown"""
        self.logger.info(f"开始处理来自 {input_dir} 的OCR文件...")
        ocr_files_data = self.read_ocr_files(input_dir)
        
        if not ocr_files_data:
            self.logger.warning("未找到OCR文件!")
            return
        
        self.logger.info(f"找到 {len(ocr_files_data)} 个OCR文件，开始清理空格...")
        # 预处理内容，删除不必要的空格
        processed_contents = self.preprocess_ocr_contents(ocr_files_data)
        
        self.logger.info("空格清理完成，开始合并页面...")
        merged_content = self.merge_ocr_contents(processed_contents)
        
        # 写入合并后的内容到Markdown文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(merged_content)
        
        self.logger.info(f"合并完成! 结果已保存至 {output_file}")



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
SESSIONS = {}
DPI=200

# --- M1/MPS-specific Helpers from m1.py -------------------------------------
def pil_from_page(page, dpi=DPI):
    #pix = page.get_pixmap(dpi=dpi)
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

def run_inference(model, inputs, max_new_tokens=15360):
    with torch.inference_mode():
        return model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=None, do_sample=False)

def decode_output(out_ids, inputs, processor):
    trimmed_ids = [o[len(iids):] for iids, o in zip(inputs["input_ids"], out_ids)]
    return processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

def extract_and_save_images(md_content, base_path, page_num):
    """
    Extracts base64 embedded images from markdown, saves them as files,
    and replaces the base64 URI with a file path.
    """
    # 修正后的正则表达式，正确捕获图片类型和base64数据
    img_pattern = re.compile(r'!\[(.*?)\]\(data:image(?:\/(\w+))?;base64,([^)]+)\)')
    matches = list(img_pattern.finditer(md_content))
    
    print(len(matches))
    for i, match in enumerate(matches):
        alt_text = match.group(1)  # 第一个捕获组是alt文本
        img_type = match.group(2)  # 第二个捕获组是图片类型(如果有)
        b64_data = match.group(3)  # 第三个捕获组是base64数据
        
        img_bytes = base64.b64decode(b64_data)
        
        if not img_type:
            try:
                with Image.open(io.BytesIO(img_bytes)) as img:
                    img_type = img.format.lower()
            except Exception as e:
                print(f"Could not determine image type with Pillow: {e}")
                img_type = 'jpeg'  # Default to jpeg if detection fails

        # Create a unique filename
        img_filename = f"page_{page_num:03d}_image_{i+1}.{img_type}"
        img_path = os.path.join(base_path, img_filename)
        
        with open(img_path, "wb") as f:
            f.write(img_bytes)
            
        # Replace the base64 URI with the new file path in the markdown
        md_content = md_content.replace(match.group(0), f"![{alt_text}]({img_filename})")
        
    return md_content
                                    
# --- Gradio App Helpers (Adapted from demo_gradio.py) -----------------------
def get_initial_session_state():
    return {
        'processing_results': { 'temp_dir': None, 'session_id': None, 'status': 'idle' },
        'pdf_cache': {
            "images": [], "current_page": 0, "total_pages": 0,
            "file_type": None, "is_parsed": False, "results": [], "original_sizes": []
        },
        'original_filename': None
    }

def create_session_dir(session_id):
    """Creates a dedicated, persistent directory for session outputs in ./gradio_output."""
    output_root = "gradio_output"
    os.makedirs(output_root, exist_ok=True)
    session_dir = os.path.join(output_root, f"session_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

def load_file_for_preview(file_path, session_state):
    pdf_cache = session_state['pdf_cache']
    if not file_path or not os.path.exists(file_path):
        return None, "<div id='page_info_box'>0 / 0</div>", session_state
    
    session_state['original_filename'] = os.path.basename(file_path)
    
    file_ext = os.path.splitext(file_path)[1].lower()
    pages = []
    original_sizes = []
    try:
        if file_ext == '.pdf':
            doc = fitz.open(file_path)
            for p in doc:
                pages.append(pil_from_page(p, dpi=DPI))
                original_sizes.append((p.rect.width, p.rect.height))
            pdf_cache["file_type"] = "pdf"
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            img = Image.open(file_path)
            pages = [img]
            original_sizes = [(img.width, img.height)]
            pdf_cache["file_type"] = "image"
        else:
            return None, "<div id='page_info_box'>Unsupported</div>", session_state
    except Exception as e:
        return None, f"<div id='page_info_box'>Load failed: {e}</div>", session_state
    
    pdf_cache.update({"images": pages, "original_sizes": original_sizes, "current_page": 0, "total_pages": len(pages), "is_parsed": False, "results": []})
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

def get_or_create_session(token=None):
    if token and token in SESSIONS:
        return token, SESSIONS[token]
    token = uuid.uuid4().hex
    SESSIONS[token] = get_initial_session_state()
    return token, SESSIONS[token]

# --- Core Local Inference Function ------------------------------------------
def process_file_local(session_token, test_image_input, file_input, prompt_mode):
    # --- Hardcoded API and Model for automatic merging ---
    API_BASE = "http://127.0.0.1:1234/v1/"
    MODEL_NAME = "openai/gpt-oss-20b"

    if not session_token:
        session_token, _ = get_or_create_session()

    session_state = SESSIONS[session_token]
    session_state['processing_results']['status'] = 'processing'

    if not MODEL or not PROCESSOR:
        session_state['processing_results']['status'] = 'error'
        yield None, "Error: Model not loaded. Check console for errors.", "", "", gr.update(value=None), None, "", session_state, session_token, ""
        return

    input_path = file_input if file_input else test_image_input
    if not input_path:
        session_state['processing_results']['status'] = 'error'
        yield None, "Please upload a file or select an example.", "", "", gr.update(value=None), None, "", session_state, session_token, ""
        return

    # Setup session directory
    session_id = session_token
    session_dir = create_session_dir(session_id)
    abs_session_dir = os.path.abspath(session_dir)
    print(f"[INFO] Session results are being saved to: {abs_session_dir}")
    session_state['processing_results'].update({'temp_dir': session_dir, 'session_id': session_id})

    pdf_cache = session_state['pdf_cache']
    images_to_process = pdf_cache.get("images", [])
    original_sizes = pdf_cache.get("original_sizes", [])
    if not images_to_process: # If file wasn't previewed, load it now
        _, _, session_state = load_file_for_preview(input_path, session_state)
        images_to_process = session_state['pdf_cache'].get("images", [])
        original_sizes = session_state['pdf_cache'].get("original_sizes", [])

    prompt_text = dict_promptmode_to_prompt.get(prompt_mode, "")
    all_results = []
    all_md_content = []
    total_pages = len(images_to_process)
    
    # Initial message with output location
    initial_info_text = f"**Output Location:** `{abs_session_dir}`\n\n**Local Inference Details:**\n- Device: {DEVICE}\n- Total Pages: {total_pages}\n- Session ID: {session_id}"

    for i, img in enumerate(images_to_process):
        page_num = i + 1
        
        # Update UI to show progress before processing
        info_text = f"{initial_info_text}\n\n**Processing page {page_num}/{total_pages}...**"
        page_info_html = f"<div id='page_info_box'>{page_num} / {total_pages}</div>"
        processing_md = "\n\n---\n\n".join(all_md_content) + f"\n\n---\n\n*⏳ Processing page {page_num}...*"
        
        yield (
            img, info_text, processing_md, processing_md,
            gr.update(visible=False), page_info_html, "Processing...", session_state, session_token, ""
        )

        out_text = None
        try:
            inputs = prepare_inputs(img, prompt_text, PROCESSOR, DEVICE, DTYPE)
            out_ids = run_inference(MODEL, inputs)
            out_text = decode_output(out_ids, inputs, PROCESSOR)
            
            parsed_json = parse_json_flex(out_text)
            
            # Normalize the structure to always be a dictionary with a 'blocks' key
            if isinstance(parsed_json, list):
                page_info = {"blocks": parsed_json}
            else:
                page_info = parsed_json

            # Now, filter the blocks within the unified structure
            page_info = filter_json_blocks(page_info)

            original_width, original_height = original_sizes[i]
            #img_for_md = img.resize([0, 0, original_width, original_height])
            img_for_md = img.resize((int(original_width), int(original_height)))

            page_info.update({"page": page_num, "width": img_for_md.width, "height": img_for_md.height})
            
            md_content = layoutjson2md(img_for_md, page_info.get('blocks', []), text_key='text')
            
            current_page_result = {'json_data': page_info, 'md_content': md_content}
            all_results.append(current_page_result)
            all_md_content.append(md_content)

            # Save artifacts
            json_path = os.path.join(session_dir, f"page_{page_num:03d}.json")
            md_path = os.path.join(session_dir, f"page_{page_num:03d}.md")
            with open(json_path, "w", encoding="utf-8") as f: json.dump(page_info, f, ensure_ascii=False, indent=2)
            # split image files from markdown and save them separately
            md_content = extract_and_save_images(md_content, session_dir, page_num)
            with open(md_path, "w", encoding="utf-8") as f: f.write(md_content)

        except Exception as e:
            print(f"[ERROR] Failed to process page {page_num}: {e}, output: {out_text}")
            error_message = f"Failed to process page {page_num}: {e}, output: {out_text}"
            current_page_result = {'json_data': {'error': error_message}, 'md_content': f"## {error_message}"}
            all_results.append(current_page_result)
            all_md_content.append(f"## {error_message}")
        finally:
            # Clean up memory after each page
            if 'inputs' in locals(): del inputs
            if 'out_ids' in locals(): del out_ids
            gc.collect()
            if DEVICE == "mps": torch.mps.empty_cache()

        # Update session state after processing the page
        pdf_cache["results"] = all_results
        pdf_cache["current_page"] = i
        session_state['pdf_cache'] = pdf_cache

        # Yield the result for the current page
        combined_md = "\n\n---\n\n".join(all_md_content)
        current_page_json_str = json.dumps(current_page_result.get('json_data', {}), ensure_ascii=False, indent=2)
        
        yield (
            img, info_text, combined_md, combined_md,
            gr.update(visible=False), page_info_html, current_page_json_str, session_state, session_token, ""
        )

    # After the loop, do final updates
    pdf_cache.update({"is_parsed": True, "results": all_results, "current_page": 0})
    session_state['pdf_cache'] = pdf_cache
    session_state['processing_results']['status'] = 'completed'

    # --- Automatic Merging and Cleaning ---
    info_text = f"{initial_info_text}\n\n**✅ Processing complete. Now merging and cleaning...**"
    yield (
        images_to_process[0] if images_to_process else None, info_text, combined_md, combined_md,
        gr.update(visible=False), f"<div id='page_info_box'>1 / {total_pages}</div>",
        json.dumps(all_results[0].get('json_data', {}), ensure_ascii=False, indent=2) if all_results else "{}", 
        session_state, session_token, "*Merging...*"
    )

    unload_model()
    merged_content = merge_and_clean_pages_auto(session_state, API_BASE, MODEL_NAME)
    load_model()

    # Create zip for download
    original_filename = session_state.get('original_filename')
    if original_filename:
        zip_basename = os.path.splitext(original_filename)[0]
        download_zip_path = os.path.join(session_dir, f"{zip_basename}.zip")
    else:
        download_zip_path = os.path.join(session_dir, f"results_{session_id}.zip")

    with zipfile.ZipFile(download_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(session_dir):
            for file in files:
                if not file.endswith('.zip'):
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), session_dir))

    # Final yield to set the first page view and enable download
    final_info_text = f"{initial_info_text}\n\n**✅ Processing and merging complete.**"
    combined_md = "\n\n---\n\n".join(all_md_content)
    first_page_json_str = json.dumps(all_results[0].get('json_data', {}), ensure_ascii=False, indent=2) if all_results else "{}"
    first_image = images_to_process[0] if images_to_process else None
    
    yield (
        first_image, final_info_text, combined_md, combined_md,
        gr.update(value=download_zip_path, visible=True),
        f"<div id='page_info_box'>1 / {total_pages}</div>",
        first_page_json_str, session_state, session_token, merged_content
    )

def merge_and_clean_pages_auto(session_state, api_base, model_name):
    session_dir = session_state['processing_results'].get('temp_dir')

    if not session_dir or not os.path.exists(session_dir):
        return "Error: No processed files found to merge."

    try:
        merger = PDFMerger(api_base=api_base, model=model_name)
        
        def process_and_return_merged(merger_instance, input_dir: str, output_filename: str) -> str:
            merger_instance.logger.info(f"开始处理来自 {input_dir} 的OCR文件...")
            ocr_files_data = merger_instance.read_ocr_files(input_dir)
            
            if not ocr_files_data:
                merger_instance.logger.warning("未找到OCR文件!")
                return "No .md files found in the session directory."
            
            merger_instance.logger.info(f"找到 {len(ocr_files_data)} 个OCR文件，开始清理空格...")
            processed_contents = merger_instance.preprocess_ocr_contents(ocr_files_data)
            
            merger_instance.logger.info("空格清理完成，开始合并页面...")
            merged_content = merger_instance.merge_ocr_contents(processed_contents)
            
            output_filepath = os.path.join(input_dir, output_filename)
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(merged_content)
            
            merger_instance.logger.info(f"合并完成! 结果已保存至 {output_filepath}")
            return merged_content

        merged_content = process_and_return_merged(merger, session_dir, "merged_output.md")

        session_id = session_state['processing_results']['session_id']
        download_zip_path = os.path.join(session_dir, f"results_{session_id}.zip")
        with zipfile.ZipFile(download_zip_path, 'a', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(os.path.join(session_dir, "merged_output.md"), "merged_output.md")

        return merged_content

    except Exception as e:
        print(f"[ERROR] Failed to merge and clean pages: {e}")
        return f"An error occurred during the merge process: {e}"

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
        
        def on_load(request: gr.Request):
            token = request.query_params.get("token")
            token, session_state = get_or_create_session(token)
            return {session_token: gr.update(value=token), session_state_component: session_state}

        session_token = gr.Textbox(label="Session Token", interactive=False)
        session_state_component = gr.State()

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
                                md_output = gr.Markdown("## Click 'Parse Locally' to begin...", elem_id="markdown_output", latex_delimiters=[{"left": "$", "right": "$", "display": True}])
                            with gr.TabItem("Markdown Raw (Editable)"):
                                md_raw_output = gr.Textbox("...", label="Markdown Raw Text", lines=38, show_copy_button=True, show_label=False, interactive=True)
                            with gr.TabItem("Current Page JSON"):
                                current_page_json = gr.Textbox("...", label="Current Page JSON", lines=38, show_copy_button=True, show_label=False)
                            with gr.TabItem("Merged Output"):
                                merged_output = gr.Markdown("## Processing will automatically merge pages...", elem_id="markdown_output")
                
                download_btn = gr.DownloadButton("⬇️ Download Results", visible=False)

        # Event Handlers
        md_raw_output.change(fn=lambda x: x, inputs=md_raw_output, outputs=md_output)
        demo.load(on_load, None, [session_token, session_state_component])
        prompt_mode.change(fn=update_prompt_display, inputs=prompt_mode, outputs=prompt_display)
        file_input.upload(fn=load_file_for_preview, inputs=[file_input, session_state_component], outputs=[result_image, page_info, session_state_component])
        test_image_input.change(fn=load_file_for_preview, inputs=[test_image_input, session_state_component], outputs=[result_image, page_info, session_state_component])
        
        prev_btn.click(fn=lambda s: turn_page("prev", s), inputs=[session_state_component], outputs=[result_image, page_info, current_page_json, session_state_component])
        next_btn.click(fn=lambda s: turn_page("next", s), inputs=[session_state_component], outputs=[result_image, page_info, current_page_json, session_state_component])
        
        process_btn.click(
            fn=process_file_local,
            inputs=[session_token, test_image_input, file_input, prompt_mode],
            outputs=[result_image, info_display, md_output, md_raw_output, download_btn, page_info, current_page_json, session_state_component, session_token, merged_output]
        )

        clear_btn.click(
            fn=clear_all_data, inputs=[session_state_component],
            outputs=[file_input, test_image_input, result_image, info_display, md_output, md_raw_output, download_btn, page_info, current_page_json, session_state_component]
        )
    return demo

# --- Main Program Execution -------------------------------------------------
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
            # We don't exit here, as we might want to retry or handle it in the UI
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
        if DEVICE == "mps":
            torch.mps.empty_cache()
        print("[INFO] Model unloaded.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    """
    # --- Startup ---
    print("[INFO] Application startup: Loading model...")
    load_model()
    yield
    # --- Shutdown ---
    print("\n[INFO] Application shutdown: Cleaning up resources...")
    unload_model()
    
    session_root = "gradio_output"
    if os.path.exists(session_root):
        print(f"[INFO] Removing all session data from '{session_root}'...")
        try:
            shutil.rmtree(session_root)
            print("[INFO] Session data removed successfully.")
        except OSError as e:
            print(f"[ERROR] Failed to remove session data: {e}")
            
    print("[INFO] Cleanup complete. Exiting.")


if __name__ == "__main__":
    import signal

    def force_shutdown(signum, frame):
        print("\n[INFO] Ctrl+C detected. Forcefully shutting down.")
        sys.exit(0)

    signal.signal(signal.SIGINT, force_shutdown)

    print("[INFO] Starting local M1 Gradio demo...")
    if not os.path.exists(MODEL_DIR):
        print(f"[ERROR] Model directory not found at '{MODEL_DIR}'. Please download the model first.")
        sys.exit(1)

    app = FastAPI(lifespan=lifespan)

    @app.get("/api/job/{token}")
    def get_job_status(token: str):
        if token not in SESSIONS:
            raise HTTPException(status_code=404, detail="Job not found")
        return SESSIONS[token]['processing_results']

    @app.get("/")
    def root():
        return RedirectResponse(url="/gradio")

    demo = create_gradio_interface()
    app = gr.mount_gradio_app(app, demo, path="/gradio")

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    print(f"[INFO] Launching Gradio interface on http://0.0.0.0:{port}/gradio")
    uvicorn.run(app, host="0.0.0.0", port=port)
