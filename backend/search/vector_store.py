"""
Vector Store for RAG: FAISS-based implementation
- Stores chunked text and their embeddings
- Supports adding, saving, loading, and semantic search
"""
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

class VectorStore:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2', index_path=None):
        self.model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.texts = []  # List of (chunk_text, metadata)
        self.index_path = index_path or 'data/vector.index'
        self.meta_path = self.index_path + '.meta.pkl'
        self.dimension = self.model.get_sentence_embedding_dimension()
        if os.path.exists(self.index_path):
            self.load()
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

    def add_texts(self, texts_with_meta):
        """
        texts_with_meta: list of dicts with keys 'text' and optional 'meta'
        """
        new_texts = [item['text'] for item in texts_with_meta]
        embeddings = self.model.encode(new_texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        self.index.add(embeddings)
        self.texts.extend(texts_with_meta)

    def search(self, query, top_k=5):
        query_emb = self.model.encode([query]).astype('float32')
        D, I = self.index.search(query_emb, top_k)
        results = []
        for idx in I[0]:
            if idx < len(self.texts):
                results.append(self.texts[idx])
        return results

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.texts, f)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, 'rb') as f:
            self.texts = pickle.load(f)
