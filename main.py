from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer cannot be found in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
""")

loader = PyPDFLoader('data/Introduction-to-AI-and-Basic-Concepts.pdf')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap  = 50
)

chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory='./chroma_db'
)

retreiver = vector_db.as_retriever(
    search_kwargs = {"k":3}
)

question = "Which field contributed Bayes Rule to AI?"

chunks = retreiver.invoke(question)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
)


docs = retreiver.invoke(question)

print(f"Retrieved {len(docs)} documents\n")

for i, doc in enumerate(docs):
    print(f"========== Chunk {i+1} ==========")
    print(doc.page_content)
    print()

rag_chain = (
    {
        "context": retreiver | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

results = rag_chain.invoke(question)

print(results)