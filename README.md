#  ICD-11 RAG Medical Assistant

An **AI-powered ICD-11 Medical Assistant** that retrieves official WHO ICD-11 definitions, synonyms, and related info using **semantic search** + **local LLM reasoning**.

✅ ICD-11 official WHO dataset  
✅ Synonym-aware semantic retrieval  
✅ Fast local inference with **Ollama LLMs**  
✅ Modern UI with clickable WHO references  

---

##  Features  

- **RAG pipeline** → Retrieves ICD-11 entries, synonyms & definitions  
- **Synonym expansion** → Understands layman queries like *“heart attack” → myocardial infarction*  
- **Vector DB with Chroma** → Efficient & persistent search  
- **Local LLM reasoning** → Runs `phi3:medium` or `mistral` with Ollama or any other local model you want to use
- **Modern UI** → Built in Gradio, shows sources  

---

##  Project Structure  

```
ICD_RAG_AGENT/
│── .venv/                 # Virtual environment
│── debug/                 # Debugging scripts (CUDA/GPU checks)
│── docs/                  # Docs (optional)
│── embeddings/            # Persistent Chroma DB (auto-created)
│── utils/                 # Utility scripts
│   ├── convert_json_to_text.py
│   ├── fetch_all_entities.py   # Fetches ICD-11 JSONs from WHO API
│   ├── get_token.py           # Retrieves WHO API token
│   ├── icd_embed_store.py
│   └── icd_text_splitter.py
│── app.py                  # Main Gradio UI
│── preprocess.py           # Prepares ICD data → ChromaDB
│── test.py                 # Simple retrieval test
│── requirements.txt
│── README.md
│── LICENSE.txt
```

 **Only these are essential for data preparation:**  
- `get_token.py` → WHO API token  
- `fetch_all_entities.py` → Fetch raw ICD JSONs  
- `preprocess.py` → Process + embed ICD JSONs  

All other utils are refactored inside `preprocess.py`.

---

##  Setup  

### 1️ Install dependencies  

```bash
git clone <repo-url>
cd ICD_RAG_AGENT
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

### 2️ Install & Run Ollama  

```bash
# Install Ollama (https://ollama.ai)
ollama serve

# Pull a small LLM (phi3:medium or mistral)
ollama pull phi3:medium
```

---

### 3️ WHO API Setup  

WHO provides **official ICD-11 REST APIs**.  

📄 **Read WHO API Docs:**  
👉 [https://icd.who.int/icdapi](https://icd.who.int/icdapi)  

1. Get a WHO API key → store in `.env` or pass as arg  
2. Run:  

```bash
python utils/get_token.py
```

This will fetch & cache an **access token**.  

---

### 4️ Fetch ICD-11 JSONs  

```bash
python utils/fetch_all_entities.py
```

This will download **all ICD-11 foundation entities** as JSON under `docs/` or `icd_data/`.  

---

### 5️ Preprocess ICD Data  

Once JSONs are fetched:  

```bash
python preprocess.py
```

This will:  
- Extract **title + synonyms + definitions**  
- Generate **embeddings**  
- Store into `embeddings/` as a **persistent ChromaDB**  

---

### 6️ Run the App  

```bash
python app.py
```

Access **http://localhost:7860**  

---

##  How it Works  

1. **User asks:** *What is a heart attack?*  
2. Query → normalized → **synonym expansion**  
3. Vector search in Chroma → finds **Myocardial infarction** entry  
4. ICD-11 context → passed to Ollama LLM for strict answer  
5. UI shows answer + **WHO ICD official link**  

---

##  Example  

**Q:** *What is a heart attack?*  
**A:**  

> **Myocardial infarction** is necrosis of heart muscle due to ischemia…  
>  
> 📚 ICD Reference → [Myocardial infarction](https://icd.who.int/browse/2025-01/foundation/en#123456)  

---

##  Improving Retrieval  

-  Layman → medical mapping  
-  Synonym-aware embeddings  
-  Confidence-based “Did you mean?”  
-  Hybrid BM25 + vectors for rare cases  

---

##  Roadmap  

- [ ] Add **auto-abbreviation handling** (MI → Myocardial infarction)  
- [ ] Multi-language ICD translations  
- [ ] Deploy with **Docker + GPU acceleration**  

---

##  License  

MIT  

---

### TL;DR  

```bash
# 1. Fetch WHO API data
python utils/get_token.py
python utils/fetch_all_entities.py

# 2. Preprocess & build Chroma
python preprocess.py

# 3. Serve local LLM
ollama serve &
ollama pull phi3:medium

# 4. Launch UI
python app.py
```
User asks: What is a heart attack?

Query → normalized → synonym expansion

Vector search in Chroma → finds Myocardial infarction entry

ICD-11 context → passed to Ollama LLM for strict answer

UI shows answer + WHO ICD official link

# Example
Q: What is a heart attack?
A:
Myocardial infarction is necrosis of heart muscle due to ischemia…

 ICD Reference → Myocardial infarction

# Improving Retrieval
 Layman → medical mapping

 Synonym-aware embeddings

 Confidence-based “Did you mean?”

 Hybrid BM25 + vectors for rare cases

# Roadmap
 
 Add auto-abbreviation handling (MI → Myocardial infarction)

 Multi-language ICD translations

 Deploy with Docker + GPU acceleration

 # License

MIT



