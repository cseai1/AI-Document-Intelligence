# RAG ChatBot: Document Intelligence with AI

A polished Retrieval-Augmented Generation (RAG) project that allows users to upload documents and chat with them using AI.

This repository is designed as a  project with a production-like interface, multi-session handling, document persistence, and vector search.

## ✅ Key Features

- Upload documents in multiple formats: `PDF`, `DOCX`, `TXT`, `CSV`, `PNG`, `JPG`
- Index documents using FAISS and Google Gemini embeddings
- Multi-session workspace with isolated chat history per document
- Persistent recovery of uploaded files, vectors, and chat history
- Clean Streamlit frontend with modern interface
- Safe local vector reload for trusted index files

## 🚀 Stable Launch Command

Run the stable application entrypoint:

```powershell
python -m streamlit run app/main.py
```

## 📦 Setup

1. Create a `.env` file in the repository root.
2. Add your Google API key:

```text
GOOGLE_API_KEY="YOUR_KEY_HERE"
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Start the app:

```powershell
python -m streamlit run app/main.py
```

## 📁 Repository Structure

- `app/main.py` — stable entrypoint for the refined UI
- `app/app_refined.py` — the final polished application code
- `app/app_advanced.py` — alternate version with advanced dashboard layout
- `app/app_ultra_visual.py` — alternate visual UI version
- `app/utils/prepare_vectordb.py` — document loading and FAISS vector creation
- `app/utils/chatbot.py` — retrieval and model response logic
- `requirements.txt` — main dependencies
- `requirements-vision.txt` — extra OCR/image requirements
- `.gitignore` — ignores secrets and generated storage

## 💡 Notes for Git

- Do not commit `.env`
- Generated data directories are ignored:
  - `docs/`
  - `faiss_indexes/`
  - `Vector_DB - Documents/`
- Keep only `app/main.py` as the primary run target for users

## 📌 Best Practices

- Use `app/main.py` when sharing or deploying
- Keep API keys secret and stored in `.env`
- If you want only the clean production-ready app, focus on `app/app_refined.py`

## 🎯 Project Value

This project is suitable for a final-year CSE AI submission because it solves a real-world document understanding problem with:

- a full-stack AI flow
- vector search persistence
- multi-document chat
- production-oriented usability


