import os
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------
# CONFIG
# ----------------------------
TEXT_DIR = "docs"         # folder where your .txt files are saved
OUTPUT_FILE = "split_chunks.txt"  # optional debug output
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_text_documents(text_dir):
    """Load all .txt files into LangChain Document objects"""
    docs = []
    for filename in os.listdir(text_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(text_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:  # only add non-empty files
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": filename}
                    ))
                else:
                    print(f"[SKIPPED] Empty file: {filename}")
    return docs

def split_documents(docs):
    """Split large docs into smaller chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)

if __name__ == "__main__":
    print(" Loading ICD-11 text files...")
    docs = load_text_documents(TEXT_DIR)
    print(f" Loaded {len(docs)} text docs from '{TEXT_DIR}'")

    print(" Splitting into chunks...")
    chunks = split_documents(docs)
    print(f" Split into {len(chunks)} chunks!")

    # Optional: Save chunks for debugging
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i+1} ({chunk.metadata['source']}) ---\n")
            f.write(chunk.page_content + "\n\n")

    print(f" Saved debug chunks to {OUTPUT_FILE}")
