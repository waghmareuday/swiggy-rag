import time
import threading
import gradio as gr

vector_store = None
init_status_msg = "Initializing system automatically..."
init_lock = threading.Lock()


def _background_init():
    global vector_store, init_status_msg
    try:
        from src.config import PDF_PATH
        from src.document_processor import process_document
        from src.vector_store import build_or_load_vector_store, vector_store_exists

        with init_lock:
            if vector_store is not None:
                return

            if vector_store_exists():
                vector_store = build_or_load_vector_store()
                init_status_msg = "Vector store loaded from disk. Ready to answer questions!"
            else:
                if not PDF_PATH.exists():
                    init_status_msg = (
                        f"PDF not found at:\n{PDF_PATH}\n\n"
                        "Please place the Swiggy Annual Report PDF in the data/ folder."
                    )
                    return
                chunks = process_document()
                vector_store = build_or_load_vector_store(chunks)
                init_status_msg = f"Document processed: {len(chunks)} chunks indexed. Ready to answer questions!"
    except Exception as e:
        init_status_msg = f"Error during initialization: {str(e)}"


def initialize_system():
    global vector_store

    if vector_store is not None:
        return init_status_msg

    try:
        from src.config import PDF_PATH
        from src.document_processor import process_document
        from src.vector_store import build_or_load_vector_store, vector_store_exists

        if vector_store_exists():
            vector_store = build_or_load_vector_store()
            return "Vector store loaded from disk. Ready to answer questions!"
        else:
            if not PDF_PATH.exists():
                return (
                    f"PDF not found at:\n{PDF_PATH}\n\n"
                    "Please place the Swiggy Annual Report PDF in the data/ folder."
                )
            chunks = process_document()
            vector_store = build_or_load_vector_store(chunks)
            return f"Document processed: {len(chunks)} chunks indexed. Ready to answer questions!"
    except Exception as e:
        return f"Error during initialization: {str(e)}"


def answer_query(question, history):
    global vector_store

    if vector_store is None:
        return "Please click Initialize System first before asking questions.", ""

    if not question.strip():
        return "Please enter a question.", ""

    try:
        from src.rag_pipeline import ask_question

        start_time = time.time()
        answer, sources = ask_question(vector_store, question)
        elapsed = round(time.time() - start_time, 2)
        return f"{answer}\n\n*Response time: {elapsed}s*", sources
    except Exception as e:
        return f"Error: {str(e)}", ""


def clear_all():
    return "", "", ""


TITLE = """
# Swiggy Annual Report — RAG Q&A System
### Ask questions about the Swiggy Annual Report and get accurate, context-grounded answers.
"""

EXAMPLES = [
    "What is Swiggy's total revenue for FY 2023-24?",
    "Who are the board members of Swiggy?",
    "What are Swiggy's key business segments?",
    "What is Swiggy's employee strength?",
    "What is the CSR expenditure of Swiggy?",
    "What are the major risks mentioned in the report?",
    "What is Swiggy's net loss for the fiscal year?",
    "Who is the CEO of Swiggy?",
]


def build_ui():
    with gr.Blocks(title="Swiggy RAG Q&A") as app:

        gr.Markdown(TITLE)

        with gr.Row():
            with gr.Column(scale=3):
                init_btn = gr.Button("Initialize System", variant="primary", size="lg")
            with gr.Column(scale=7):
                init_status = gr.Textbox(
                    label="System Status",
                    interactive=False,
                    value="Click Initialize System to start...",
                )

        init_btn.click(fn=initialize_system, outputs=init_status)

        gr.Markdown("---")

        with gr.Row():
            question_input = gr.Textbox(
                label="Ask a Question",
                placeholder="Type your question about the Swiggy Annual Report...",
                lines=2,
                scale=4,
            )
            with gr.Column(scale=1):
                ask_btn = gr.Button("Ask", variant="primary", size="lg")
                clear_btn = gr.Button("Clear", size="sm")

        answer_output = gr.Textbox(
            label="Answer",
            interactive=False,
            lines=8,
        )

        with gr.Accordion("Supporting Context (Source Chunks)", open=False):
            source_output = gr.Markdown(
                label="Sources",
                elem_classes=["source-box"],
            )

        gr.Markdown("### Example Questions")
        gr.Examples(examples=EXAMPLES, inputs=question_input, label="")

        ask_btn.click(
            fn=answer_query,
            inputs=[question_input, gr.State([])],
            outputs=[answer_output, source_output],
        )

        question_input.submit(
            fn=answer_query,
            inputs=[question_input, gr.State([])],
            outputs=[answer_output, source_output],
        )

        clear_btn.click(
            fn=clear_all,
            outputs=[question_input, answer_output, source_output],
        )

        gr.Markdown(
            "---\n"
            "**Built with** LangChain | FAISS | Google Gemini | HuggingFace | Gradio\n\n"
            "Answers are generated strictly from the Swiggy Annual Report. "
            "The system will not hallucinate or use external knowledge."
        )

    return app


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 10000))
    is_cloud = os.environ.get("RENDER", "")

    print(f"Starting server on port {port} (cloud={bool(is_cloud)})")

    threading.Thread(target=_background_init, daemon=True).start()
    print("Background initialization started...")

    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=not is_cloud,
        inbrowser=not is_cloud,
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="amber"),
        css=".gradio-container { max-width: 1000px; margin: auto; } .source-box { max-height: 400px; overflow-y: auto; }",
    )
