import os
from dotenv import load_dotenv
import chromadb
import google.generativeai as genai
import time

# Load environment variables from .env file
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=gemini_key)

# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="./db/chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(name=collection_name)


# =================================
# === For initial setup -- Uncomment (below) all for the first run, and then comment it all out ===
# =================================

# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents


# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


# Load documents from the directory
directory_path = "./data/new_articles"
documents = load_documents_from_directory(directory_path)

# Split the documents into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})


# Function to generate embeddings using Gemini API with retry logic
def get_gemini_embedding(text, task_type="retrieval_document", max_retries=3):
    for attempt in range(max_retries):
        try:
            response = genai.embed_content(
                model="models/gemini-embedding-2",
                content=text,
                task_type=task_type
            )
            print("==== Generating embeddings... ====")
            return response["embedding"]
        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                wait_time = 7  # Wait 7 seconds before retrying
                print(f"Rate limited. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                raise


# Generate embeddings for the document chunks
for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = get_gemini_embedding(doc["text"], task_type="retrieval_document")


# Upsert documents with embeddings into Chroma
for doc in chunked_documents:
    print("==== Inserting chunks into db ====")
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["text"]],
        embeddings=[doc["embedding"]],  
    )

# === End of the initial setup -- Uncomment all for the first run, and then comment it all out ===
# =================================


# Function to query documents according to the question
def query_documents(question, n_results=2):
    query_embedding = get_gemini_embedding(question, task_type="retrieval_query")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks


# Function to generate a standardized response from Gemini LLM 
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    return response.text


question = "tell me about databricks acquisition of ai"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print("==== Answer ====")
print(answer)