import chromadb

from chromadb.utils import embedding_functions  
default_ef = embedding_functions.DefaultEmbeddingFunction()

#creating embeddings
# default_ef = embedding_functions.DefaultEmbeddingFunction()
# name = "Purushoth"
# emb = default_ef(name)
# print(emb)  

documents = [
    {"id": "doc1", "text": "hello world!"},
    {"id": "doc2", "text": "how are you?"},
    {"id": "doc3", "text": "This is the third document."},
    {"id": "doc4", "text": "Microsoft is the best company document."}
]

chroma_client = chromadb.PersistentClient(path="./db/chroma_db")  #Specify the path for persistent storage

# creates collection using the default embedding function
collection = chroma_client.get_or_create_collection(
    name="MY_COLLECTIONS", embedding_function=default_ef
)

#inserting all the doc 1 by 1 to convert them unto embeddings and store in the collection
for doc in documents:
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["text"]]
    )

# Searches for the 3 documents most similar to "hello World!" using embeddings.
results = collection.query(
    query_texts=["hello World!"],
    n_results=3
)

query_text = "hello World!"

for idx, documents in enumerate(results["documents"][0]):
    doc_id = results["ids"][0][idx]
    distance = results["distances"][0][idx]
    print(f"Document ID: {doc_id}, Distance: {distance}, Text: {documents}")


