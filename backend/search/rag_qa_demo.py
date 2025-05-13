"""
End-to-end RAG QA Demo: Retrieve + Generate
- Retrieves top-k relevant chunks from vector DB
- Uses a transformer generator (GPT-2) to answer the user's question
"""
from backend.search.vector_store import VectorStore
from backend.search.generator_openai import OpenAIGenerator

if __name__ == "__main__":
    vs = VectorStore(index_path='data/vector.index')
    gen = OpenAIGenerator(model_name='gpt-3.5-turbo')
    print("\nRAG QA Demo (type 'exit' to quit):")
    while True:
        query = input("\nEnter your question: ").strip()
        if query.lower() in ("exit", "quit"): break
        retrieved = vs.search(query, top_k=5)
        print("\nRetrieved context:")
        for i, chunk in enumerate(retrieved, 1):
            print(f"[{i}] {chunk['text'][:200]}{'...' if len(chunk['text'])>200 else ''}")
        answer = gen.generate(query, retrieved)
        print(f"\nGenerated Answer:\n{answer}\n")
