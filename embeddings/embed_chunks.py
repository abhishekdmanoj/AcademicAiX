import numpy as np

def embed_chunks(chunks, model):
    """
    Embeds syllabus text chunks using the provided model.
    """
    embeddings = model.encode(chunks, show_progress_bar=True)
    return np.array(embeddings)