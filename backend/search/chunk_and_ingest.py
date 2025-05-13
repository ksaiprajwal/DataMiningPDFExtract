"""
Chunk extracted text and ingest into the FAISS vector store.
- Reads extracted text from a file (e.g., data/sample_extracted.txt)
- Splits text into overlapping chunks
- Adds chunks to the vector store
- Saves the vector index
"""
import os
from backend.search.vector_store import VectorStore
import re

def chunk_text(text, chunk_size=300, overlap=50):
    """
    Splits text into overlapping chunks.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks

def ingest_extracted_file(txt_path, vector_store_path='data/vector.index'):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Extracted text file not found: {txt_path}")
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Split by page markers if present
    pages = re.split(r'--- Page \d+ ---', content)
    all_chunks = []
    for page_num, page_text in enumerate(pages):
        page_text = page_text.strip()
        if not page_text:
            continue
        for chunk in chunk_text(page_text):
            all_chunks.append({'text': chunk, 'meta': {'page': page_num+1}})
    vs = VectorStore(index_path=vector_store_path)
    vs.add_texts(all_chunks)
    vs.save()
    print(f"Ingested {len(all_chunks)} chunks into vector store at {vector_store_path}")

if __name__ == "__main__":
    txt_path = os.path.join('data', 'sample_extracted.txt')
    ingest_extracted_file(txt_path)
