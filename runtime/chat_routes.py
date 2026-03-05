import os
import json
import numpy as np
import faiss
import pickle
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from groq import Groq

router = APIRouter(prefix="/chat", tags=["Chat"])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METADATA_PATH = os.path.join(PROJECT_ROOT, "data", "university_metadata.json")
CHAT_INDEX_PATH = os.path.join(PROJECT_ROOT, "vector_store", "faiss_chat.index")
CHAT_METADATA_PATH = os.path.join(PROJECT_ROOT, "vector_store", "metadata_chat.pkl")

# Load API key from environment
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"

_chat_index = None
_chat_meta = None


def load_chat_index():
    global _chat_index, _chat_meta
    if _chat_index is not None:
        return
    if not os.path.exists(CHAT_INDEX_PATH) or not os.path.exists(CHAT_METADATA_PATH):
        print("Chat index not found. Run: python -m offline_pipeline.build_chat_index")
        return
    _chat_index = faiss.read_index(CHAT_INDEX_PATH)
    with open(CHAT_METADATA_PATH, "rb") as f:
        _chat_meta = pickle.load(f)
    print(f"Chat index loaded: {len(_chat_meta)} chunks")


load_chat_index()


class ChatRequest(BaseModel):
    message: str
    college: str = ""
    program: str = ""


def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def get_relevant_chunks(query: str, college: str = "", program: str = "", top_k: int = 5):
    if _chat_index is None or _chat_meta is None:
        print("Chat index not loaded")
        return []

    try:
        import api
        model = api.model
        if model is None:
            return []

        query_vec = model.encode([query], convert_to_numpy=True)
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

        search_k = 300 if (college or program) else top_k * 2
        distances, indices = _chat_index.search(query_vec.astype("float32"), min(search_k, len(_chat_meta)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(_chat_meta):
                continue
            meta = _chat_meta[idx]

            if college and program:
                if (meta.get("college", "").lower() != college.lower() or
                        meta.get("program", "").lower() != program.lower()):
                    continue

            results.append({
                "text": meta.get("text", ""),
                "college": meta.get("college", ""),
                "program": meta.get("program", ""),
                "score": float(dist)
            })

            if len(results) >= top_k:
                break

        return results

    except Exception as e:
        print(f"FAISS search error: {e}")
        return []


def ask_groq(prompt: str) -> str:
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq error: {e}")
        return None


def build_prompt(question: str, chunks: list, college: str = "", program: str = "") -> str:
    context = "\n\n---\n\n".join([
        f"[{c['college']} - {c['program']}]\n{c['text']}"
        for c in chunks
    ])
    scope = f" for {college} - {program}" if college and program else ""
    return f"""You are AcademicAiX, an AI assistant helping students understand university programs and syllabi.

Use ONLY the syllabus content provided below to answer the question. Do not make up information.
If the answer is not in the provided content, say so clearly.

SYLLABUS CONTENT{scope}:
{context}

STUDENT QUESTION: {question}

Provide a clear, helpful answer based strictly on the syllabus content above.
Be concise but complete. Use bullet points where appropriate."""


def extractive_answer(chunks: list) -> str:
    if not chunks:
        return "I couldn't find relevant information for your question."

    college = chunks[0].get("college", "")
    program = chunks[0].get("program", "")

    all_text = " ".join([c["text"].replace("\n", " ") for c in chunks])
    sentences = [s.strip() for s in all_text.split(".") if len(s.strip()) > 30]

    seen = set()
    good = []
    for s in sentences:
        key = s[:40].lower()
        if key not in seen:
            seen.add(key)
            good.append(s)
        if len(good) >= 5:
            break

    if not good:
        return "I couldn't find relevant information for your question."

    lines = [f"**{college} - {program}**", ""]
    for s in good:
        lines.append(f"• {s}.")

    return "\n".join(lines)


def get_entrance_info(college: str, program: str) -> str:
    try:
        metadata = load_json(METADATA_PATH)
        for entry in metadata:
            if (entry.get("college", "").lower() == college.lower() and
                    entry.get("program", "").lower() == program.lower()):
                exams = entry.get("entrance_exams", [])
                if exams:
                    lines = [f"**Entrance Exams for {program} at {college}:**"]
                    for exam in exams:
                        line = f"• {exam['name']}"
                        if exam.get("website"):
                            line += f" - {exam['website']}"
                        lines.append(line)
                    return "\n".join(lines)
        return None
    except Exception as e:
        print(f"Metadata lookup error: {e}")
        return None


def extract_program_from_message(message: str, metadata: list) -> tuple:
    msg_lower = message.lower()
    for entry in metadata:
        college = entry.get("college", "")
        program = entry.get("program", "")
        if college.lower() in msg_lower and program.lower() in msg_lower:
            return college, program
        if college.lower() in msg_lower:
            return college, program
    return "", ""


def detect_intent(message: str) -> str:
    msg = message.lower()
    if any(w in msg for w in ["entrance", "exam", "gate", "jee", "neet", "cuet", "cat",
                               "admission", "eligibility", "need", "require", "qualify", "how to apply"]):
        return "entrance"
    if any(w in msg for w in ["syllabus", "subject", "course", "topic", "curriculum",
                               "what do i study", "modules", "cover", "covers", "covered"]):
        return "syllabus"
    if any(w in msg for w in ["find", "recommend", "suggest", "help me", "which program", "best program"]):
        return "discover"
    return "general"


@router.post("")
async def chat(req: ChatRequest):
    message = req.message.strip()
    college = req.college.strip()
    program = req.program.strip()

    if not message:
        return JSONResponse({
            "success": False,
            "answer": "Please ask a question.",
            "source": "none"
        })

    intent = detect_intent(message)

    if not college or not program:
        metadata = load_json(METADATA_PATH)
        detected_college, detected_program = extract_program_from_message(message, metadata)
        if detected_college:
            college = detected_college
        if detected_program:
            program = detected_program

    if intent == "entrance" and college and program:
        exam_info = get_entrance_info(college, program)
        if exam_info:
            return JSONResponse({
                "success": True,
                "answer": exam_info,
                "source": "metadata",
                "intent": intent
            })

    chunks = get_relevant_chunks(message, college, program, top_k=5)

    if not chunks and (college or program):
        print(f"No filtered chunks found for {college} - {program}, broadening search")
        chunks = get_relevant_chunks(message, top_k=5)

    if not chunks:
        return JSONResponse({
            "success": True,
            "answer": "I couldn't find relevant information for your question. Try asking about a specific topic like 'data structures' or 'machine learning'.",
            "source": "none",
            "intent": intent
        })

    prompt = build_prompt(message, chunks, college, program)
    groq_response = ask_groq(prompt)

    if groq_response:
        return JSONResponse({
            "success": True,
            "answer": groq_response,
            "source": "groq",
            "intent": intent,
            "chunks_used": len(chunks)
        })

    answer = extractive_answer(chunks)
    return JSONResponse({
        "success": True,
        "answer": answer,
        "source": "extractive",
        "intent": intent,
        "chunks_used": len(chunks)
    })
