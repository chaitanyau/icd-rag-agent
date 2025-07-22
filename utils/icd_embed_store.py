from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

#  1. Load all text files
docs_path = Path("docs")
documents = []
for file in docs_path.glob("*.txt"):
    loader = TextLoader(file, encoding="utf-8")
    documents.extend(loader.load())

print(f" Loaded {len(documents)} ICD-11 text documents")

#  2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(documents)
print(f" Split into {len(chunks)} chunks")

#  3. Use a BETTER embedding model
# Best retrieval models (pick one):
# "intfloat/e5-large-v2"  → Great for RAG
# "BAAI/bge-large-en-v1.5" → Very strong on MTEB leaderboard
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

#  4. Store in Chroma
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="embeddings"
)

print(f" Embedded {len(chunks)} chunks → saved to 'embeddings/'")
