from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
print(" Model on:", m.device)
print(m.encode(["Hello GPU!"], convert_to_tensor=True).shape)
