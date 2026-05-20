import os

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import chromadb
from dotenv import load_dotenv 
from chromadb.utils import embedding_functions
from openai import OpenAI
import httpx
from typing import List

# Create httpx client with SSL verification disabled
http_client = httpx.Client(verify=False)

# Custom embedding function using OpenAI with disabled SSL verification
class CustomOpenAIEmbedding:
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key, http_client=http_client)
        self.model_name = model_name
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            input=input,
            model=self.model_name
        )
        return [item.embedding for item in response.data]
    
    def embed_query(self, input: str) -> List[float]:
        """Embed a single query string"""
        response = self.client.embeddings.create(
            input=input,
            model=self.model_name
        )
        return response.data[0].embedding
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return self(documents)
    
    def name(self) -> str:
        return "custom-openai-embedding"  




load_dotenv()  # Load environment variables from .env file


# Get the OpenAI API key from environment variables
open_api_key = os.getenv("CHROMA_OPENAI_API_KEY")

# Create custom OpenAI embedding function with SSL verification disabled
openai_ef = CustomOpenAIEmbedding(
    api_key=open_api_key,
    model_name="text-embedding-3-small"
)
# name = "Purushoth"
# emb = default_ef(name)
# print(emb)  

#inserting all the doc 1 by 1 to convert them unto embeddings and store in the collection
# for doc in documents:
#     collection.upsert(
#         ids=[doc["id"]],
#         documents=[doc["text"]]
#     )

chroma_client = chromadb.PersistentClient(path="./db/chroma_db")  #Specify the path for persistent storage

# creates collection using the OpenAI embedding function
collection = chroma_client.get_or_create_collection(
    name="my-story", embedding_function=openai_ef
)
documents = [
    {"id": "doc1", "text": "Generative AI is a type of artificial intelligence that creates new content such as text, images, audio, or code."},
    {"id": "doc2", "text": "Large Language Models are AI systems trained on vast amounts of text to understand and generate human-like language."},
    {"id": "doc3", "text": "Prompt engineering is the process of designing effective inputs to guide AI models toward useful and accurate outputs."},
    {"id": "doc4", "text": "Tokenization is the method of breaking text into smaller units called tokens so AI models can process language."},
    {"id": "doc5", "text": "Fine-tuning is the practice of training a pre-trained AI model on specific data to improve performance for a targeted task."},
    {"id": "doc6", "text": "Embeddings are numerical representations of text or data that capture semantic meaning for search and similarity tasks."},
    {"id": "doc7", "text": "Retrieval-Augmented Generation combines information retrieval with text generation to produce more accurate and context-aware responses."},
    {"id": "doc8", "text": "Hallucination in Generative AI refers to a model producing incorrect or fabricated information that appears believable."},
    {"id": "doc9", "text": "Inference is the process of using a trained AI model to generate predictions or outputs from new input data."},
    {"id": "doc10", "text": "Transformer architecture is a deep learning framework widely used in modern Generative AI models for handling sequences efficiently."},
    {"id": "doc11", "text": "Multimodal AI can process and generate multiple types of data, including text, images, audio, and video."},
    {"id": "doc12", "text": "Responsible AI focuses on fairness, transparency, privacy, safety, and accountability in AI system design and use."}
]


# Searches for the 3 documents most similar to "hello World!" using embeddings.
results = collection.query(
    query_texts=["hello World!"],
    n_results=3
)

query_text = "find the document related to Generative AI"

for idx, documents in enumerate(results["documents"][0]):
    doc_id = results["ids"][0][idx]
    distance = results["distances"][0][idx]
    print(f"Document ID: {doc_id}, Distance: {distance}, Text: {documents}")


