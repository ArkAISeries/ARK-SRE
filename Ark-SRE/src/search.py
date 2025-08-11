import sys
import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "ark_sre_db"
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection("ark_sre")

embedder = SentenceTransformer('all-MiniLM-L6-v2')

query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter your search query: ")
query_emb = embedder.encode(query).tolist()

results = collection.query(query_embeddings=[query_emb], n_results=3)
print("\n🔍 Search results for:", query)
if results["documents"]:
    for doc in results["documents"][0]:
        print("-", doc.split("A:", 1)[1].strip())
else:
    print("⚠️ No results found.")
