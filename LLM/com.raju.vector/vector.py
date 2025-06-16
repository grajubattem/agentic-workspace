#pip install faiss-cpu openai langchain numpy langhchain-community
 # Generate Embedding
 
# import openai
import numpy as np
import openai
# OpenAI API Key (Replace with yours)
#openai.api_key =

from dotenv import load_dotenv
 
load_dotenv()

# Sample documents
documents = [
    "Machine learning helps computers learn from data.",
    "Neural networks are inspired by the human brain.",
    "FAISS is a library for fast similarity search.",
    "Natural language processing enables AI to understand text.",
    "Transformers are deep learning models used in NLP."
]

# Convert documents into embeddings
def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

    # Store embeddings
embeddings = np.array([get_embedding(doc) for doc in documents])

#  Storing embedding in FAISS
 
 
import faiss
 
# Get vector dimensions (from first embedding)
d = embeddings.shape[1]
 
# Initialize FAISS index
index = faiss.IndexFlatL2(d)  # L2 distance (Euclidean)
index.add(embeddings)  # Add all document embeddings
 
print(f"✅ Stored {index.ntotal} document embeddings in FAISS.")

# Performing Similarity Search
 
query_text = "Explain how AI understands text."
 
# Get query embedding
query_embedding = np.array([get_embedding(query_text)])
 
# Search for similar embeddings
k = 2  # Number of top results
distances, indices = index.search(query_embedding, k)
 
# Display results
print(" Top matching documents:")
for i in range(k):
    print(f"{i+1}. {documents[indices[0][i]]} (Distance: {distances[0][i]:.2f})")

