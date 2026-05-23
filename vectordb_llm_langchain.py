import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Use Gemini model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Use Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

persist_directory = "./db/chroma_db_real_world"

# Try to load existing database FIRST (skip embedding step!)
if os.path.exists(persist_directory):
    print("Loading existing database...")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    print("Creating new database...")
    
    # Load documents
    loader = DirectoryLoader(
        path="./data/new_articles", 
        glob="**/*.txt", 
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}  
    )
    document = loader.load()

    # Text splitting - larger chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
    texts = text_splitter.split_documents(document)
    print(f"Number of chunks: {len(texts)}")

    # Create vector database (only if it doesn't exist)
    vectordb = Chroma.from_documents(
        documents=texts, embedding=embeddings, persist_directory=persist_directory
    )

# Now we can query the Chroma object
retriever = vectordb.as_retriever()

# Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

system_prompt = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say that you don't know.
Use three sentences maximum and keep the answer concise.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

# Build RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

response = rag_chain.invoke("talk about databricks news.")
print(response)