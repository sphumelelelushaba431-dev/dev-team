"""Central configuration. Every setting can be overridden with an environment variable."""
import os
from pathlib import Path

try:  # optional .env support
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KB_DIR = BASE_DIR / "knowledge_base"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Complaint categories (departments) used for routing
CATEGORIES = ["Academic", "Finance", "ICT", "Facilities", "Hostel"]

# Dataset: the Kaggle "University Students Complaints & Reports" CSV if present,
# otherwise the bundled synthetic sample.
COMPLAINTS_CSV = os.getenv("COMPLAINTS_CSV", str(DATA_DIR / "university_complaints.csv"))
SAMPLE_CSV = str(DATA_DIR / "sample_complaints.csv")

# ---- Embeddings / vector database -------------------------------------------
# "sbert" = sentence-transformers (downloads the model once)
# "lsa"   = TF-IDF + truncated SVD (fully offline fallback)
# "auto"  = sbert if installed, else lsa
EMBEDDER = os.getenv("EMBEDDER", "auto")
SBERT_MODEL = os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "120"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "30"))
TOP_K = int(os.getenv("TOP_K", "4"))
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.15"))

# ---- Deep learning classifier -----------------------------------------------
# "bert" = fine-tune a transformer (DistilBERT by default; needs torch + transformers)
# "mlp"  = TF-IDF + multilayer perceptron neural network (offline fallback)
# "auto" = bert if torch and transformers are installed, else mlp
DL_BACKEND = os.getenv("DL_BACKEND", "auto")
BERT_MODEL = os.getenv("BERT_MODEL", "distilbert-base-uncased")
BERT_EPOCHS = int(os.getenv("BERT_EPOCHS", "3"))

# ---- Large language model -----------------------------------------------------
# "anthropic" | "openai" | "groq" (free tier) | "ollama" (free, local) | "hf" (local
# Hugging Face model) | "none" | "auto"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "google/flan-t5-base")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))

# ---- Routing and guardrails ---------------------------------------------------
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
MEMORY_TURNS = int(os.getenv("MEMORY_TURNS", "6"))

# ---- Storage ------------------------------------------------------------------
# If FIREBASE_CREDENTIALS points to a service-account JSON file, Firestore is
# used; otherwise a local SQLite database.
FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS", "")
SQLITE_PATH = os.getenv("SQLITE_PATH", str(BASE_DIR / "campusvoice.db"))

# ---- Web app ------------------------------------------------------------------
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")

# ---- Attitude dataset evaluation (RAG vs LLM) -----------------------------------
CV_FOLDS = int(os.getenv("CV_FOLDS", "5"))
RAG_K = int(os.getenv("RAG_K", "7"))

# ---- Optional channels --------------------------------------------------------
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM = os.getenv("TWILIO_FROM", "")
TWILIO_NOTIFY_TO = os.getenv("TWILIO_NOTIFY_TO", "")
