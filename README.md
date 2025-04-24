# The Assistant to The Office Chatbot

Welcome to **The Assistant to The Office Chatbot**, a creative AI project that brings fictional characters to life through dynamic, in-character conversations. Built around *The Office (US)*, this chatbot allows users to chat with TV show characters like Michael Scott or Dwight Schrute and watch how their responses evolve across different seasons.

This project is designed as a full-stack AI product using Natural Language Processing (NLP), character modeling, and optionally custom-trained language models. The goal is to allow freeform conversation with a character, not just retrieving old quotes — but actually generating new dialogue in their personality and context.

---

## 🚀 Main Features
- Choose a character (e.g., Michael Scott)
- Select their development stage (e.g., Season 1–3, Season 4–6)
- Chat with them using natural language
- Receive context-aware replies that reflect their tone, language, and knowledge from that era

---

## 🧠 Tech Stack (To Be Expanded)
- Python (NLP, data prep, modeling)
- Hugging Face Transformers (model training & inference)
- FAISS / ChromaDB (for semantic search — optional)
- Streamlit or Gradio (UI layer)
- JupyterLab (for EDA, modeling, fine-tuning)

---

## 📦 Project Structure (Coming Soon)
```
📁 data/                # Dialogue dataset
📁 notebooks/           # EDA, preprocessing, fine-tuning
📁 app/                 # Web app UI & inference code
📄 requirements.txt     # Required Python libraries
📄 README.md            # Project overview
```

---

## 🧰 Set up your Environment
The added [requirements file](requirements.txt) contains all libraries and dependencies we need to execute Pandas and Numpy.

### **macOS** – run the following:
```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### **WindowsOS** – run the following:

**For PowerShell CLI:**
```powershell
pyenv local 3.11.3
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**For Git-Bash CLI:**
```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** If you encounter an error with pip upgrade:
```bash
python.exe -m pip install --upgrade pip
```

---

## ✅ Current Dependencies
To get started, install the dependencies listed below. These libraries cover data handling, visualization, machine learning, and the core NLP functionality we'll use to build the chatbot.

```
jupyterlab==3.6.3
seaborn==0.12.2
numpy==1.24.3
pandas==2.0.1
scikit-learn==1.2.2
transformers==4.40.0
datasets==2.18.0
sentence-transformers==2.2.2
streamlit==1.32.0
```

These will be updated as the project evolves and new tools are integrated.

