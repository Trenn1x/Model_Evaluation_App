from fastapi import FastAPI, Request
from transformers import pipeline
import faiss
import numpy as np
import time

app = FastAPI()

# Initialize a Hugging Face pipeline for text generation
model_pipeline = pipeline("text-generation", model="distilgpt2")

# Initialize FAISS index for vector storage and retrieval (RAG functionality)
dimension = 768  # Example embedding dimension
index = faiss.IndexFlatL2(dimension)

# Mock database of texts and embeddings
text_database = []
embedding_database = []

@app.get("/")
def read_root():
    """Root endpoint to confirm app is running."""
    return {"message": "AI Model Evaluation App"}

@app.post("/generate_text/")
async def generate_text(request: Request):
    """
    Generates text using an LLM and measures response latency.
    Request Body:
        - text: Input text to generate a response from.
    Response:
        - input_text: The original input.
        - response_text: Generated response from the model.
        - latency: Time taken to generate the response.
    """
    data = await request.json()
    input_text = data.get("text", "")
    
    # Start monitoring time
    start_time = time.time()

    # Generate a response using the LLM
    output = model_pipeline(input_text, max_length=50, num_return_sequences=1, truncation=True)
    response_text = output[0]["generated_text"]

    # End monitoring time
    end_time = time.time()
    latency = end_time - start_time

    return {"input_text": input_text, "response_text": response_text, "latency": latency}

@app.post("/add_to_index/")
async def add_to_index(request: Request):
    """
    Adds text and its embedding to the FAISS index for retrieval purposes.
    Request Body:
        - text: The text to add to the index.
    Response:
        - message: Confirmation message.
        - text: The original text that was indexed.
    """
    data = await request.json()
    text = data.get("text", "")
    
    # Generate a mock embedding and add it to FAISS index
    embedding = np.random.rand(1, dimension).astype("float32")  # Using random embedding as placeholder
    index.add(embedding)
    text_database.append(text)
    embedding_database.append(embedding)

    return {"message": "Text added to index", "text": text}

@app.post("/search/")
async def search(request: Request):
    """
    Searches the FAISS index for the closest match to the input query text.
    Request Body:
        - query: Text to find similar matches for.
    Response:
        - query_text: The original query text.
        - closest_text: The closest matching text in the index.
        - distance: Distance score indicating similarity.
    """
    data = await request.json()
    query_text = data.get("query", "")
    
    # Generate a mock embedding for the query
    query_embedding = np.random.rand(1, dimension).astype("float32")

    # Perform a search on the FAISS index
    distances, indices = index.search(query_embedding, k=1)

    # Retrieve the closest text match
    closest_text = text_database[indices[0][0]] if indices[0][0] < len(text_database) else "No match found"

    return {"query_text": query_text, "closest_text": closest_text, "distance": float(distances[0][0])}

