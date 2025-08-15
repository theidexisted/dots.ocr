

import os
import argparse
from dots_ocr.parser import DotsOCRParser

def combine_markdown_files(results, output_path):
    """
    Combines markdown content from multiple result objects into a single file.

    Args:
        results (list): A list of result dictionaries from the parser.
        output_path (str): The path to the final combined markdown file.
    """
    full_markdown_content = []
    print(f"Combining {len(results)} pages into a single markdown file...")

    # Sort results by page number to ensure correct order
    results.sort(key=lambda x: x.get('page_no', 0))

    for result in results:
        md_path = result.get('md_content_path')
        if md_path and os.path.exists(md_path):
            with open(md_path, 'r', encoding='utf-8') as f:
                full_markdown_content.append(f.read())
        else:
            print(f"Warning: Markdown file not found for page {result.get('page_no')}")

    with open(output_path, 'w', encoding='utf-8') as f:
        # Add a separator between pages for clarity
        f.write('\n\n---\n\n'.join(full_markdown_content))
    
    print(f"Successfully created combined markdown file at: {output_path}")

def main():
    """
    Main function to parse a PDF and generate a single combined markdown file.
    """
    parser = argparse.ArgumentParser(description="Parse a PDF to a single Markdown file using DotsOCRParser.")
    parser.add_argument("input_pdf", help="The absolute path to the input PDF file.")
    parser.add_argument("--output_dir", default="./output", help="Directory to save intermediate and final files.")
    parser.add_argument("--ip", default="localhost", help="IP address of the inference server.")
    parser.add_argument("--port", type=int, default=8000, help="Port of the inference server.")
    args = parser.parse_args()

    # --- 1. Initialize the Parser ---
    # This assumes you have a vLLM or compatible server running.
    # For local HF models, you would set use_hf=True.
    dots_ocr_parser = DotsOCRParser(
        ip=args.ip,
        port=args.port,
        output_dir=args.output_dir
    )

    # --- 2. Parse the PDF file ---
    # The parse_file method handles both PDF and image files, creating a subdirectory for the results.
    # It returns a list of dictionaries, one for each page.
    print(f"Starting PDF parsing for: {args.input_pdf}")
    results = dots_ocr_parser.parse_file(
        input_path=args.input_pdf,
        prompt_mode="prompt_layout_all_en" # Use the general-purpose prompt
    )

    # --- 3. Combine the results into a single Markdown file ---
    if results:
        pdf_filename = os.path.splitext(os.path.basename(args.input_pdf))[0]
        combined_md_path = os.path.join(args.output_dir, f"{pdf_filename}_full.md")
        combine_markdown_files(results, combined_md_path)
    else:
        print("Parsing did not return any results.")

if __name__ == "__main__":
    main()

