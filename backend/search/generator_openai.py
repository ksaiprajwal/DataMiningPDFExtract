"""
OpenAI-based Generator for RAG: Uses OpenAI's GPT-3.5-turbo (or similar) via API
- Reads key from OPENAI_API_KEY env var or .env file
"""
import os
import openai
from dotenv import load_dotenv
load_dotenv()


api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError('OPENAI_API_KEY environment variable not set.')
client = openai.OpenAI(api_key=api_key)

class OpenAIGenerator:
    def __init__(self, model_name='gpt-3.5-turbo'):
        self.model_name = model_name

    def generate(self, query, retrieved_chunks, max_tokens=256):
        context = '\n'.join([chunk['text'] for chunk in retrieved_chunks])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers based only on the provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
