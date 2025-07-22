"""
Convert WHO ICD-11 JSON → clean text → Chroma vector store with rich metadata.

Features:
    ✅ Extract ICD code, title, synonyms, definition, browser URL
    ✅ Skip entries with missing definitions (optional)
    ✅ Chunk text with RecursiveCharacterTextSplitter
    ✅ Embed with BAAI/bge-large-en-v1.5 (high MTEB retrieval quality)
    ✅ Store in Chroma with rich metadata (icd_code, source_url, file, chunk pos)
    ✅ Progress bar for embedding (batch mode)
    ✅ Incremental persistence (resume-safe)

Usage:
    python preprocess.py \
        --json_dir docs/raw_json \
        --txt_dir docs/clean_txt \
        --index_dir embeddings \
        --skip_missing_defs \
        --model BAAI/bge-large-en-v1.5 \
        --chunk_size 800 \
        --chunk_overlap 100
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict

import torch
from tqdm import tqdm

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ NEW import


# ------------------------- Helper Functions ---------------------------------

def load_json(fp: Path) -> dict:
    with fp.open("r", encoding="utf-8-sig") as f:  # ✅ handle UTF-8 BOM
        return json.load(f)

def extract_code_from_id(raw_id: str) -> str:
    """Return last path token of @id as ICD code."""
    return raw_id.rstrip("/").split("/")[-1]

def entry_to_text(entry: dict, code: str) -> str:
    """Serialize a JSON entry to plain text block."""
    title = entry.get("title", {}).get("@value", f"Untitled ICD Entry {code}")
    synonyms = [s.get("label", {}).get("@value", "") for s in entry.get("synonym", [])]
    definition = entry.get("definition", {}).get("@value", "")
    url = entry.get("browserUrl", "")

    parts: List[str] = [f"Title: {title}"]
    if synonyms:
        parts.append("Synonyms: " + "; ".join(synonyms))
    if definition:
        parts.append("Definition: " + definition)
    if url:
        parts.append("SourceURL: " + url)
    parts.append(f"ICDCode: {code}")
    return "\n".join(parts)

def detect_device() -> str:
    """Detect GPU if available for embeddings."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------- Main Preprocessing Routine --------------------------

def build_index(json_dir: Path,
                txt_dir: Path,
                index_dir: Path,
                model: str,
                chunk_size: int,
                chunk_overlap: int,
                skip_missing_defs: bool,
                batch_size: int = 50):

    txt_dir.mkdir(parents=True, exist_ok=True)

    code2url: Dict[str, str] = {}
    cleaned_files: List[Path] = []
    skipped_missing_defs = 0
    total_json = 0

    # -------- Pass 1: JSON → cleaned .txt ----------------
    for jf in json_dir.glob("*.json"):
        total_json += 1
        try:
            data = load_json(jf)
            code = extract_code_from_id(data.get("@id", jf.stem))
            browser_url = data.get("browserUrl", "")
            code2url[code] = browser_url

            definition = data.get("definition", {}).get("@value", "")
            if skip_missing_defs and not definition:
                skipped_missing_defs += 1
                continue  # skip this entry entirely

            txt_block = entry_to_text(data, code)
            fname = re.sub(r"[^\w\- ]", "", f"{code}_{data.get('title',{}).get('@value','')}")[:80]
            out_fp = txt_dir / f"{fname}.txt"
            out_fp.write_text(txt_block, encoding="utf-8")
            cleaned_files.append(out_fp)

        except Exception as e:
            print(f"[skip] {jf.name}: {e}")

    print(f"\n[✓] Processed {total_json} JSON entries")
    if skip_missing_defs:
        print(f"[⚠] Skipped {skipped_missing_defs} entries with missing definitions")
    print(f"[✓] Saved {len(cleaned_files)} text entries → {txt_dir}")

    # -------- Pass 2: Chunk text ----------------
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs: List[Document] = []
    for tfile in cleaned_files:
        code = tfile.stem.split("_")[0]
        url = code2url.get(code, "")
        raw_text = tfile.read_text(encoding="utf-8")
        for pos, chunk in enumerate(splitter.split_text(raw_text)):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "icd_code": code,
                    "browser_url": url,
                    "source_file": tfile.name,
                    "position": pos,
                }
            ))

    print(f"[✓] Total chunks: {len(docs)} – preparing to embed with {model}")

    # -------- Pass 3: Embed in batches with progress bar --------
    device = detect_device()
    print(f"🚀 Using device: {device.upper()}")
    emb = HuggingFaceEmbeddings(model_name=model, model_kwargs={"device": device})

    vectordb = None
    for i in tqdm(range(0, len(docs), batch_size), desc="🔄 Embedding Chunks"):
        batch_docs = docs[i:i + batch_size]
        if vectordb is None:
            # create first batch
            vectordb = Chroma.from_documents(
                documents=batch_docs,
                embedding=emb,
                persist_directory=str(index_dir)
            )
        else:
            # add subsequent batches
            vectordb.add_documents(batch_docs)
        vectordb.persist()  # persist incrementally

    print(f"\n[✓] Embedded {len(docs)} chunks and saved → {index_dir}")


# ------------------------------ CLI -----------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json_dir", type=Path, required=True, help="Folder with ICD-11 raw JSONs")
    p.add_argument("--txt_dir", type=Path, required=True, help="Output folder for cleaned .txt")
    p.add_argument("--index_dir", type=Path, required=True, help="Chroma DB persist dir")
    p.add_argument("--model", default="BAAI/bge-large-en-v1.5",
                   help="Embedding model (default: BAAI/bge-large-en-v1.5)")
    p.add_argument("--chunk_size", type=int, default=800)
    p.add_argument("--chunk_overlap", type=int, default=100)
    p.add_argument("--skip_missing_defs", action="store_true",
                   help="Skip ICD entries without definitions entirely")
    p.add_argument("--batch_size", type=int, default=50,
                   help="Number of chunks to embed per batch (default: 50)")
    args = p.parse_args()

    build_index(
        args.json_dir,
        args.txt_dir,
        args.index_dir,
        args.model,
        args.chunk_size,
        args.chunk_overlap,
        args.skip_missing_defs,
        args.batch_size
    )
