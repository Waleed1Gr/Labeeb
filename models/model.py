import whisper
import faiss
from sentence_transformers import SentenceTransformer

# ============== LOAD MODELS ==============
print("⚙️ تحميل Whisper...")
whisper_model = whisper.load_model("medium")

print("⚙️ تحميل SentenceTransformer...")
sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

dimension = 384
tasks = []
embeddings = []
index = faiss.IndexFlatL2(dimension)
