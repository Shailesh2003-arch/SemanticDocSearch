import os
from pathlib import Path
from dotenv import load_dotenv
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams

load_dotenv()



pdf_path = Path(__file__).parent/ "Node Js.pdf"
# Load file into the current program
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load() # This will give you documents page-by-page
# print(docs)
# print(len(docs))

# Split the docs into smaller chunks...

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(documents=docs)
print(f"Total chunks after splitting: {len(chunks)}")


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# vector store
# --- Qdrant setup ---
client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API") ,timeout=60)

# Recreate collection to match embedding size (384 for MiniLM)
collection_name = "SemanticDocs"
vector_size = 384
distance = "Cosine"

client.recreate_collection(
    collection_name=collection_name,
    vectors_config = VectorParams(
        size=vector_size,
        distance=distance
    )
)

vector_store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embedding_model)
vector_store.add_documents(chunks, batch_size=20)

print("✅ All chunks inserted into Qdrant successfully")