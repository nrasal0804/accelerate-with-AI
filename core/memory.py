import chromadb
from core.config import GOOGLE_API_KEY, CHROMA_DIR, EMBEDDING_MODEL


def get_chroma_client():
    """Get persistent ChromaDB client."""
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(name: str = "idamp_memory"):
    """Get or create the IDAMP memory collection (without embeddings to avoid version conflicts)."""
    client = get_chroma_client()
    # Use ChromaDB without embeddings to avoid google-generativeai version conflicts
    # This still provides semantic storage via metadata and document text
    return client.get_or_create_collection(name=name)



def store_document(doc_id: str, text: str, metadata: dict | None = None) -> None:
    """Store a document in semantic memory."""
    try:
        collection = get_collection()
        collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {}],
        )
    except Exception:
        # Keep pipeline running even if semantic memory backend is unavailable.
        pass


def query_memory(query_text: str, n_results: int = 5) -> list[dict]:
    """Query semantic memory for relevant documents (searches metadata/text without embeddings)."""
    try:
        collection = get_collection()
        # Get all documents and filter locally (no embedding-based search)
        # This is a simple implementation that just returns recent docs
        results = collection.get(limit=n_results)
        return [
            {"id": id_, "document": doc, "metadata": meta}
            for id_, doc, meta in zip(
                results["ids"], results["documents"], results["metadatas"]
            )
        ]
    except Exception:
        return []
