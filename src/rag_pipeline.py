from typing import List, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import GOOGLE_API_KEY, GEMINI_MODEL, TEMPERATURE, MAX_OUTPUT_TOKENS, TOP_K
from src.vector_store import similarity_search


SYSTEM_PROMPT = """You are a precise and helpful AI assistant that answers questions about the Swiggy Annual Report.

STRICT RULES:
1. Answer ONLY based on the provided context from the Swiggy Annual Report.
2. Do NOT use any external knowledge or make assumptions beyond what is in the context.
3. If the context does not contain enough information to answer the question, respond with:
   "I don't have enough information in the Swiggy Annual Report to answer this question."
4. Always be factual, concise, and accurate.
5. When citing numbers, dates, or specific data, quote them exactly as they appear in the context.
6. If the question is unrelated to Swiggy or the annual report, politely redirect the user.

CONTEXT FROM SWIGGY ANNUAL REPORT:
{context}
"""

QA_PROMPT_TEMPLATE = """
{system_prompt}

USER QUESTION: {input}

ANSWER (based strictly on the above context):"""


def get_llm() -> ChatGoogleGenerativeAI:
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY not found. Please set it in your .env file. "
            "Get a free key at: https://aistudio.google.com/apikey"
        )

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        convert_system_message_to_human=True,
    )
    return llm


def format_source_context(documents: List[Document]) -> str:
    sources = []
    for i, doc in enumerate(documents, 1):
        page = doc.metadata.get("page", "N/A")
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        snippet = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content

        sources.append(
            f"**Source {i}** | Page: {page} | Chunk: {chunk_id}\n"
            f"{snippet}\n"
        )
    return "\n---\n".join(sources)


def ask_question(
    vector_store: FAISS,
    question: str,
) -> Tuple[str, str]:
    relevant_docs = similarity_search(vector_store, question)
    context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

    llm = get_llm()

    filled_prompt = QA_PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT.format(context=context),
        input=question,
    )

    response = llm.invoke(filled_prompt)
    answer = response.content
    sources = format_source_context(relevant_docs)

    return answer, sources


if __name__ == "__main__":
    from src.vector_store import build_or_load_vector_store
    from src.document_processor import process_document

    try:
        vs = build_or_load_vector_store()
    except (FileNotFoundError, ValueError):
        chunks = process_document()
        vs = build_or_load_vector_store(chunks)

    question = "What is Swiggy's total revenue?"
    answer, sources = ask_question(vs, question)
    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{answer}")
    print(f"\nSources:\n{sources}")
