from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

model_local = ChatOllama(model="llama3")

urls = [
    "https://ollama.com/",
    "https://ollama.com/blog/windows-preview",
    "https://ollama.com/blog/openai-compatibility",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
docs_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents = docs_splits,
    collection_name = "web-based-rag-chroma",
    embedding = embeddings.OllamaEmbeddings(model="nomic-embed-text"),
)

retriever = vectorstore.as_retriever()

# Before applying rag system
print("Before RAG is applied: \n")
before_rag_request = "What is {topic}"
before_rag_prompt = ChatPromptTemplate.from_template(before_rag_request)
before_rag_chain = before_rag_prompt | model_local | StrOutputParser()

print(before_rag_chain.invoke({"topic": "Ollama"}))

# After applying rag system
print("After RAG is applied: \n")
after_rag_request = "Answer the question based on the following context: {context} Question: {question}"

after_rag_prompt = ChatPromptTemplate.from_template(after_rag_request)
after_rag_chain = (
    {"context":retriever, "question": RunnablePassthrough()} | after_rag_prompt | model_local | StrOutputParser()
)

print(after_rag_chain.invoke("What is Ollama?"))