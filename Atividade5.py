# Encoding windows
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA

sys.stdout.reconfigure(encoding='utf-8')

# CARREGA O DOCUMENTO DPF E O SEPARA EM PÁGINAS
pdf_path = './Redes-neurais.pdf'
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

# DIVIDE O DOCUMENTO EM CHUNKS
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=50,
    length_function=len,
)

texts = text_splitter.split_documents(pages)

# MOSTRA A QUANTIDADE DE CHUNKS NO TEXTO
print(f"Quantidade de chunks: {len(texts)} ")

# CRIA O BANCO DE DADOS VETORIAIS
db = FAISS.from_documents(texts, OllamaEmbeddings(model="mxbai-embed-large"))

query = "Qual o tema do texto?"
docs = db.similarity_search(query)
#print(docs[0].page_content)

# CRIANDO O MODELO LLM
model = OllamaLLM(model="llama3.2:latest")

# CRIAÇÃO DO RETRIEVER UTILIZANDO A BASE DE DADO VETORIAL
retriever = db.as_retriever(search_kwargs={"k": 5})

qa_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type="stuff")

response = qa_chain.invoke(query)
print("QA Response:", response)