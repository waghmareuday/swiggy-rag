import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, PDF_PATH


def load_pdf(pdf_path: Path = PDF_PATH) -> List[Document]:
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found at {pdf_path}. "
            "Please place the Swiggy Annual Report PDF in the data/ folder "
            "and name it 'swiggy_annual_report.pdf'."
        )

    print(f"Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    print(f"  Loaded {len(pages)} pages")
    return pages


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text


def clean_documents(documents: List[Document]) -> List[Document]:
    cleaned = []
    for doc in documents:
        cleaned_content = clean_text(doc.page_content)
        if len(cleaned_content) > 50:
            doc.page_content = cleaned_content
            cleaned.append(doc)
    print(f"  Cleaned documents: {len(cleaned)} pages retained")
    return cleaned


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    print(f"  Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def process_document(pdf_path: Path = PDF_PATH) -> List[Document]:
    print("=" * 60)
    print("DOCUMENT PROCESSING PIPELINE")
    print("=" * 60)

    pages = load_pdf(pdf_path)
    cleaned_pages = clean_documents(pages)
    chunks = chunk_documents(cleaned_pages)

    print(f"\nDocument processing complete: {len(chunks)} chunks ready")
    print("=" * 60)
    return chunks


if __name__ == "__main__":
    chunks = process_document()
    print(f"\nSample chunk (index 0):\n{chunks[0].page_content[:500]}")
    print(f"\nMetadata: {chunks[0].metadata}")
