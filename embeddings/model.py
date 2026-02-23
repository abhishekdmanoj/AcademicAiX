from sentence_transformers import SentenceTransformer

_model = None

def load_embedding_model():
    global _model

    if _model is None:
        print("🔄 Loading embedding model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Model loaded")

    return _model