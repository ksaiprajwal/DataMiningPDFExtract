"""
Generator module for RAG: Uses a pre-trained transformer model (e.g., GPT-2) to generate answers
from retrieved context + user query.
"""
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

class Generator:
    def __init__(self, model_name='distilgpt2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # Force CPU-only for compatibility on macOS
        self.generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, device=-1)

    def generate(self, query, retrieved_chunks, max_length=256):
        """
        Combines retrieved context and query, generates an answer.
        """
        context = '\n'.join([chunk['text'] for chunk in retrieved_chunks])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        output = self.generator(prompt, max_length=max_length, num_return_sequences=1, do_sample=True)
        return output[0]['generated_text'][len(prompt):].strip()
