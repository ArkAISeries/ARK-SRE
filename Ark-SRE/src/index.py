import os
import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "ark_sre_db"
DATA_PATH = os.path.join("data", "identity.txt")

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection("ark_sre")

# Read and parse Q&A from text file
docs, ids, metas = [], [], []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    blocks = f.read().strip().split("\n\n")
    for i, block in enumerate(blocks):
        lines = block.strip().split("\n")
        if len(lines) >= 2:
            q = lines[0].replace("Q:", "").strip()
            a = lines[1].replace("A:", "").strip()
            if q and a:
                text = f"Q: {q} A: {a}"
                docs.append(text)
                ids.append(f"qa-{i}")
                metas.append({"question": q, "answer": a})

# Embed and store
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embs = embedder.encode(docs, convert_to_numpy=True).tolist()
collection.add(documents=docs, embeddings=embs, metadatas=metas, ids=ids)

print(f"✅ Ark SRE indexing complete! Stored {len(docs)} Q&A pairs.")
