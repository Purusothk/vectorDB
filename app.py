import chromadb
from chromadb.utils import embedding_functions

# default embedding fctns. and we can change the embedding functions
default_ef = embedding_functions.DefaultEmbeddingFunction()

# for chromadb this is the well suited
embedding_functions.ONNXMiniLM_L6_V2



chroma_client = chromadb.Client()

collection_name = "test_collection"
collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=default_ef)

documents = [
    {"id": "doc1", "text": "hello world!"},
    {"id": "doc2", "text": "how are you?"},
    {"id": "doc3", "text": "This is the third document."}
]

for doc in documents:
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["text"]]
    )


query_text = "hello World!"

results = collection.query(
    query_texts=[query_text],
    n_results=3
)
 

for idx, documents in enumerate(results["documents"][0]):
    doc_id = results["ids"][0][idx]
    distance = results["distances"][0][idx]
    print(f"Document ID: {doc_id}, Distance: {distance}, Text: {documents}")

# looping through these results doc id and doc distances
# we are checking similarity
