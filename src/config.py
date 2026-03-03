import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
PDF_PATH = DATA_DIR / "swiggy_annual_report.pdf"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 5

TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 1024
