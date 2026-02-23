import faiss
import pickle

def load_syllabus_index():
    index = faiss.read_index("vector_store/faiss_syllabus.index")

    with open("vector_store/metadata_syllabus.pkl", "rb") as f:
        metadata = pickle.load(f)

    return index, metadata