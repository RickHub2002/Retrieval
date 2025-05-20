# 🔍 Sistema de Recuperação de Informações com LangChain + FAISS + Ollama

Este projeto é um exemplo prático de como criar um sistema de **perguntas e respostas baseado em um PDF**, usando **embeddings**, **banco vetorial** e **LLM**.

> 📄 O conteúdo usado como base foi um PDF sobre Redes Neurais.

---

## 🧠 O que ele faz?

- Lê um PDF e divide o conteúdo em partes menores (chunks).
- Gera **embeddings vetoriais** para cada parte usando o modelo `mxbai-embed-large`.
- Armazena esses vetores no banco de dados vetorial **FAISS**.
- Usa um LLM (`llama3.2:latest`, via Ollama) para responder perguntas com base no conteúdo mais relevante do documento.
- Implementa um pipeline com o **RetrievalQA** da LangChain, que:
  - Transforma a pergunta do usuário em embeddings.
  - Busca os vetores mais similares no banco FAISS.
  - Entrega essas informações ao LLM gerar uma resposta contextualizada.

---

## 🚀 Como rodar

### 1. Clone o repositório

```bash
git clone https://github.com/RickHub2002/Retrieval
```

### 3. Instale as dependências

```bash
pip install langchain langchain-community langchain-ollama faiss-cpu
```

> ⚠️ Certifique-se também de ter o [Ollama](https://ollama.com/) instalado e rodando localmente, com o modelo `llama3.2` já baixado:

```bash
ollama run llama3.2
```

---

## 🧪 Exemplos de perguntas

```python
query = "Qual o tema principal do texto?"
response = qa_chain.invoke(query)
print("Resposta:", response)
```

---

## 📌 Observações

- É possível ajustar o tamanho dos chunks (`chunk_size`) para melhorar a granularidade da busca.
- O número de resultados retornados (`k` no `retriever`) também influencia a qualidade da resposta.
- O modelo `llama3.2` roda localmente via Ollama, então **não depende de APIs externas**.