import os
import fitz  # PyMuPDF
import json
from PIL import Image
import io

def extract_figures_from_pdf(pdf_path, output_dir='data/figures'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    figure_metadata = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        images = page.get_images(full=True)
        if not images:
            continue
        for img_idx, img_info in enumerate(images, 1):
            xref = img_info[0]
            base_name = f"page{page_num+1}_img{img_idx}"
            img_path = os.path.join(output_dir, f"{base_name}.png")
            pix = fitz.Pixmap(doc, xref)
            try:
                if pix.colorspace is None:
                    print(f"Skipped image on page {page_num+1}, index {img_idx} (no colorspace)")
                    continue
                # Convert to RGB and get PNG bytes
                if pix.n != 3:
                    pix_converted = fitz.Pixmap(fitz.csRGB, pix)
                    img_bytes = pix_converted.tobytes("png")
                    pix_converted = None
                else:
                    img_bytes = pix.tobytes("png")
                # Check for solid color image using PIL
                pil_img = Image.open(io.BytesIO(img_bytes))
                colors = pil_img.getcolors(maxcolors=256*256)
                if colors and len(colors) == 1:
                    print(f"Skipped solid color image on page {page_num+1}, index {img_idx}")
                    continue
                # Save only if not single color
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
            finally:
                pix = None  # Free memory
            # Optionally, get bbox (PyMuPDF 1.18+)
            bbox = img_info[5] if len(img_info) > 5 else None
            figure_metadata.append({
                'page': page_num+1,
                'img_index': img_idx,
                'img_path': img_path,
                'xref': xref,
                'bbox': bbox
            })
            print(f"Saved figure from page {page_num+1} as {img_path}")
    # Save metadata
    meta_path = os.path.join(output_dir, 'figures_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(figure_metadata, f, indent=2)
    print(f"Total figures found: {len(figure_metadata)}. Metadata saved to {meta_path}")

if __name__ == "__main__":
    pdf_path = os.path.join('data', 'sample.pdf')
    extract_figures_from_pdf(pdf_path)
