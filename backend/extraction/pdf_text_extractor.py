import fitz  # PyMuPDF
import os
import re

def clean_text(text):
    """Clean and preprocess extracted text."""
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s\.,;:!?()\[\]"\'-]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdf(pdf_path, output_txt_path=None):
    """
    Extracts text from a PDF file page by page, cleans it, and optionally saves to a .txt file.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    all_text = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        cleaned = clean_text(text)
        all_text.append({
            'page': page_num + 1,
            'text': cleaned
        })
        print(f"[Page {page_num + 1}] {cleaned[:100]}...")  # Preview first 100 chars

    if output_txt_path:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            for entry in all_text:
                f.write(f"\n--- Page {entry['page']} ---\n{entry['text']}\n")
        print(f"Extracted text saved to {output_txt_path}")

    return all_text

if __name__ == "__main__":
    # Placeholder PDF path (replace with your actual PDF later)
    pdf_path = os.path.join('data', 'sample.pdf')
    output_txt_path = os.path.join('data', 'sample_extracted.txt')
    try:
        extract_text_from_pdf(pdf_path, output_txt_path)
    except Exception as e:
        print(f"Error: {e}")
