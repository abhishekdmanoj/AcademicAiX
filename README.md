AcademicAiX
Semantic Academic Intelligence Platform

AcademicAiX is a lightweight, vector-based academic decision-support system that semantically aligns student interests with university syllabi using cosine similarity. It provides ranked universities along with official syllabus links and entrance examination resources.

Designed for:

8GB RAM systems

Render free-tier deployment

Controlled academic ingestion

Version-aware syllabus management

🔎 System Overview

AcademicAiX solves the problem of keyword-based college matching by using semantic embeddings.

Instead of searching for exact keywords, the system:

Converts student interest into vector embeddings.

Compares it against precomputed syllabus embeddings.

Ranks universities based on semantic similarity and topic coverage.

Returns official syllabus links and entrance exam resources.

🏗️ Architecture

AcademicAiX follows a two-phase architecture:

1️⃣ Offline Phase (Heavy Processing)

PDF ingestion

Text extraction

Chunking

Embedding generation

Cosine FAISS index creation

Version-aware registry update

Runs locally only.

2️⃣ Runtime Phase (Lightweight Inference)

Embed user query only

FAISS cosine similarity search

University ranking aggregation

Return official syllabus + entrance links

Runs on Render.

No heavy computation happens during API calls.