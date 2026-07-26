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
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage , AIMessage

from dotenv import load_dotenv
import os

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Use ONLY the provided context to answer the question in a simple way in your own words.
Also Explain the answer.

If the answer cannot be found in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
""")

rewrite_prompt = ChatPromptTemplate.from_template("""
Given the following conversation:

{chat_history}

Rewrite the user's latest question into a standalone question.

Only rewrite it.
Do not answer it.

Question:
{question}
""")

loader = PyPDFLoader('data/Introduction-to-AI-and-Basic-Concepts.pdf')
docs = loader.load()

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 40
)

chunks = r_splitter.split_documents(docs)

print(len(chunks))

embeddings = HuggingFaceEmbeddings(
    model_name = 'BAAI/bge-small-en-v1.5'
)

persist_directory = "./vector_store"

if not os.path.exists(persist_directory):
    print("Creating vector database...")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

else:
    print("Loading existing vector database...")

    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )


retriever = vector_db.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 3,
        "fetch_k" : 10
    }
)

llm = ChatGoogleGenerativeAI(
    model= "gemini-3.6-flash"
)

rag_chain = (
    {
        'context': retriever | format_docs,
        'question': RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

rewrite_chain = (
    rewrite_prompt
    | llm 
    | StrOutputParser()
)

chat_history = []

while True:

    option = input("Enter option run or exit?")

    if option == "exit":
        break

    question = input("Ask a question related to the Introduction to AI Basics")

    standalone_question = rewrite_chain.invoke({
        "chat_history" : chat_history,
        "question" : question
    })

    print("\nOriginal Question:", question)
    print("Standalone Question:", standalone_question)

    result = rag_chain.invoke(standalone_question)

    print("\nAI:", result)

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=result))

    print(result)

