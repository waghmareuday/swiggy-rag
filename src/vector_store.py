from pathlib import Path
from typing import List, Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import EMBEDDING_MODEL_NAME, VECTORSTORE_DIR, TOP_K, GOOGLE_API_KEY


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    print(f"Loading embedding model: {model_name}")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=GOOGLE_API_KEY,
    )
    print("  Embedding model loaded")
    return embeddings


def create_vector_store(
    chunks: List[Document],
    embeddings=None,
    persist_dir: Path = VECTORSTORE_DIR,
) -> FAISS:
    if embeddings is None:
        embeddings = get_embedding_model()

    print(f"Creating FAISS vector store from {len(chunks)} chunks...")
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    persist_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(persist_dir))
    print(f"  Vector store saved to: {persist_dir}")

    return vector_store


def load_vector_store(
    embeddings=None,
    persist_dir: Path = VECTORSTORE_DIR,
) -> FAISS:
    if embeddings is None:
        embeddings = get_embedding_model()

    if not persist_dir.exists():
        raise FileNotFoundError(
            f"No vector store found at {persist_dir}. "
            "Please run the indexing pipeline first."
        )

    print(f"Loading vector store from: {persist_dir}")
    vector_store = FAISS.load_local(
        str(persist_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("  Vector store loaded")
    return vector_store


def vector_store_exists(persist_dir: Path = VECTORSTORE_DIR) -> bool:
    return (persist_dir / "index.faiss").exists()


def similarity_search(
    vector_store: FAISS,
    query: str,
    top_k: int = TOP_K,
) -> List[Document]:
    results = vector_store.similarity_search_with_score(query, k=top_k)

    documents = []
    for doc, score in results:
        doc.metadata["similarity_score"] = round(float(score), 4)
        documents.append(doc)

    return documents


def build_or_load_vector_store(
    chunks: Optional[List[Document]] = None,
    force_rebuild: bool = False,
) -> FAISS:
    embeddings = get_embedding_model()

    if not force_rebuild and vector_store_exists():
        print("Existing vector store found. Loading from disk...")
        return load_vector_store(embeddings)

    if chunks is None:
        raise ValueError(
            "No existing vector store and no chunks provided. "
            "Please process the document first."
        )

    print("Building new vector store...")
    return create_vector_store(chunks, embeddings)


if __name__ == "__main__":
    from src.document_processor import process_document

    chunks = process_document()
    vs = create_vector_store(chunks)

    query = "What is Swiggy's revenue?"
    results = similarity_search(vs, query)
    print(f"\nQuery: {query}")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} (score: {doc.metadata.get('similarity_score')}) ---")
        print(doc.page_content[:300])
