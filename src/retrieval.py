# In retrieval phase, user gives a query and the query is again turned into the vector embedding, now this vector embedded query brings back the relevant chunks from the database, we provide that chunks to the chat model

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI


load_dotenv()

openAI_client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Need connection to the vector db so we can query...

vector_db  = QdrantVectorStore.from_existing_collection(
    url=os.getenv("QDRANT_URL"),
    collection_name = "SemanticDocs",
    embedding=embedding_model
)


# Take the user input...



# From this query you'll perform similarity search

def ask_question(user_query:str):
    # this will return you the relevant chunks...
    search_results = vector_db.similarity_search(query=user_query)
    if not search_results:
        return "❌ No relevant context found in the documents."

    context = "\n \n\n".join([
    f"Page Content: {result.page_content}\nPage Number:{result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
    for result in search_results])

    SYSTEM_PROMPT = f"""
    You are a helpful AI assistant who answers user query based on the available context retrieved from a PDF file along with page_contents and the page number.

    You should only answer the user based on the following context and navigate the user to open the right page number to know more.

    Context:
    {context}
    """

    response = openAI_client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
        {
            "role":"system", "content":SYSTEM_PROMPT  
        },
        {
            
            "role":"user", "content":user_query 
        }
        ]
    )

    return response.choices[0].message.content