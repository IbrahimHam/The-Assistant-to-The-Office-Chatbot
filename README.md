# 🧠 The Assistant to The Office Chatbot

Ever wished you could chat with characters from _The Office (US)_?

This AI-powered chatbot lets you talk to Pam, Jim, Dwight, Michael, and others — all while staying emotionally aware, context-driven, and hilariously in-character.

Built with ❤️ using LangChain, LangGraph, FAISS, HuggingFace Transformers, and powered by Groq’s blazing-fast LLaMA 3 model.

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-ready-teal)
![LangChain](https://img.shields.io/badge/langchain-v0.2+-orange)

---

## 🚀 Main Features

- Choose a character (e.g., Michael Scott)
- Emotion + sarcasm-aware responses using NLP models
- Contextual memory across chat turns
- Runs in both **console mode** and as a **FastAPI backend**
- Real dialogue pulled from the show and embedded for retrieval

---

## 🧠 Tech Stack

- **LLM**: Groq (LLaMA 3)
- **Embeddings**: SentenceTransformers (MiniLM)
- **Retrieval**: FAISS
- **Framework**: LangChain + LangGraph
- **NLP**: HuggingFace Transformers
- **Backend**: FastAPI
- **Frontend**: Lovable (React)
- **Notebooks**: Jupyter + pandas + PyTorch

---

## 🗂️ Project Structure

```
The-Assistant-to-The-Office-Chatbot/
├── backend/
│   ├── api/                     # FastAPI server
│   └── chatbot/                 # Modular chatbot logic
├── data/                        # CSVs, chunks, vectorstore
├── notebooks/                  # Data processing, feature engineering
├── scripts/archive/            # Archived prototypes + experiments
├── requirements.txt            # Final dependency list
└── README.md                   # You're here!
```

---

## 🧰 Set Up Your Environment

### macOS/Linux

```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
pyenv local 3.11.3
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (Git-Bash)

```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 💻 How to Run It

### 1. Console Chatbot

```bash
python -m backend.chatbot.console
```

Starts a conversation with Pam. Use `/switch <Character>` to change.

### 2. Web API

```bash
uvicorn backend.api.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API testing.

---

## 📝 Notebooks

- `1_data_preparation_and_feature_engineering.ipynb`: handles speaker cleaning, emotion/sarcasm tagging, and chunking into JSONL

Other archived notebooks and prototypes live in `scripts/archive/`.

---

## 🎉 Thanks

> "You miss 100% of the shots you don’t take. – Wayne Gretzky"  
> – Michael Scott
