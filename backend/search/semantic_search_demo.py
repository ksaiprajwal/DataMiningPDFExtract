"""
Demo: Semantic Search over PDF Chunks using FAISS Vector Store
- Loads the vector store
- Lets you enter a query and returns the most relevant chunks
"""
from backend.search.vector_store import VectorStore

if __name__ == "__main__":
    vs = VectorStore(index_path='data/vector.index')
    print("\nSemantic Search Demo (type 'exit' to quit):")
    while True:
        query = input("\nEnter your search query: ").strip()
        if query.lower() in ("exit", "quit"): break
        results = vs.search(query, top_k=5)
        if not results:
            print("No relevant chunks found.")
        else:
            print("\nTop relevant chunks:")
            for i, chunk in enumerate(results, 1):
                meta = chunk.get('meta', {})
                page = meta.get('page', '?')
                print(f"\n[{i}] (Page {page})\n{chunk['text'][:500]}{'...' if len(chunk['text'])>500 else ''}")
