import sys
import os
import json
import faiss
from pathlib import Path

# Add the root directory to sys.path so 'backend' can be imported
sys.path.append(os.getcwd())

# Now this import should work:
from app.services.embedding_service import EmbeddingService
from app.utils.chunking import chunk_documents

def build_and_save_index():
    print("Starting index generation...")
    
    # 1. Load your documents (ensure docs.json is in your project root)
    docs_path = Path("docs.json") 
    with open(docs_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    # 2. Chunk them
    chunks = chunk_documents(documents, chunk_size=400, overlap=50)
    print(f"Chunked into {len(chunks)} pieces.")

    # 3. Embed them
    embedding_service = EmbeddingService()
    texts = [c["text"] for c in chunks]
    embeddings = embedding_service.embed(texts)

    # 4. Build FAISS index
    dimension = embedding_service.dimension
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))

    # 5. Save the index and the chunks
    faiss.write_index(index, "vector_store.index")
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    
    print("SUCCESS: 'vector_store.index' and 'chunks.json' are now populated!")

if __name__ == "__main__":
    build_and_save_index()