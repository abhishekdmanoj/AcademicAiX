import os
import json
import faiss
import numpy as np
import pickle

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

INDEX_PATH         = os.path.join(PROJECT_ROOT, "vector_store", "faiss_syllabus.index")
METADATA_PATH      = os.path.join(PROJECT_ROOT, "vector_store", "metadata_syllabus.pkl")
CHAT_INDEX_PATH    = os.path.join(PROJECT_ROOT, "vector_store", "faiss_chat.index")
CHAT_METADATA_PATH = os.path.join(PROJECT_ROOT, "vector_store", "metadata_chat.pkl")
CHAT_CHUNK_SIZE    = 800


def append_to_index(pdf_path: str, college: str, program: str, file_path_relative: str):
    """
    Incrementally add a single new program to both FAISS indexes.
    Called when status == "new" in register_program().
    Much faster than full rebuild — only embeds the new PDF.
    """
    from embeddings.model import load_embedding_model
    from embeddings.embed_chunks import embed_chunks
    from offline_pipeline.build_syllabus_index import (
        extract_text_from_pdf, chunk_text
    )

    print(f"   📐 Appending to index: {college} - {program}")

    model = load_embedding_model()

    text   = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, max_chars=CHAT_CHUNK_SIZE)

    if not chunks:
        print(f"   ⚠ No valid chunks — index not updated for {program}")
        return False

    embeddings = embed_chunks(chunks, model)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    # ── Syllabus index (program centroid) ────────────────────────────
    centroid = np.mean(embeddings, axis=0).astype("float32").reshape(1, -1)
    faiss.normalize_L2(centroid)

    if os.path.exists(INDEX_PATH):
        syllabus_index = faiss.read_index(INDEX_PATH)
    else:
        syllabus_index = faiss.IndexFlatIP(embeddings.shape[1])

    syllabus_index.add(centroid)
    faiss.write_index(syllabus_index, INDEX_PATH)

    # Update syllabus metadata
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "rb") as f:
            syllabus_meta = pickle.load(f)
    else:
        syllabus_meta = []

    syllabus_meta.append({
        "college":   college,
        "program":   program,
        "file_path": file_path_relative
    })

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(syllabus_meta, f)

    # ── Chat index (one vector per chunk) ────────────────────────────
    if os.path.exists(CHAT_INDEX_PATH):
        chat_index = faiss.read_index(CHAT_INDEX_PATH)
    else:
        chat_index = faiss.IndexFlatIP(embeddings.shape[1])

    chat_index.add(embeddings)
    faiss.write_index(chat_index, CHAT_INDEX_PATH)

    # Update chat metadata
    if os.path.exists(CHAT_METADATA_PATH):
        with open(CHAT_METADATA_PATH, "rb") as f:
            chat_meta = pickle.load(f)
    else:
        chat_meta = []

    for i, chunk_txt in enumerate(chunks):
        chat_meta.append({
            "college":   college,
            "program":   program,
            "file_path": file_path_relative,
            "text":      chunk_txt,
            "chunk_id":  i
        })

    with open(CHAT_METADATA_PATH, "wb") as f:
        pickle.dump(chat_meta, f)

    print(f"   ✅ Index updated — {len(chunks)} chunks added")
    return True
