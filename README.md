# 🤖 Chat IA com RAG (Retrieval-Augmented Generation)

Este projeto é um assistente virtual inteligente capaz de ler documentos, entender o contexto e responder perguntas baseadas estritamente nas informações fornecidas.

Ele utiliza a técnica de **RAG (Retrieval-Augmented Generation)**, criando um banco de vetores (Vector Store) a partir dos documentos para garantir respostas precisas e contextualizadas.

## 🚀 Funcionalidades

- 📄 **Leitura de Documentos:** Suporta carregamento de arquivos de texto/PDF.
- 🧠 **Embeddings e Vetores:** Converte o texto em vetores numéricos para busca semântica.
- 🔍 **Busca Contextual:** Encontra os trechos mais relevantes do documento antes de responder.
- 💬 **Chat Interativo:** Interface para conversar com a IA sobre o conteúdo do documento.
- 💾 **Persistência de Dados:** O banco de vetores é salvo localmente na pasta `db/`, evitando reprocessamento desnecessário.

## 🛠️ Tecnologias Utilizadas

- **Python 3.13+**
- **LangChain** (Orquestração do fluxo de IA)
- **ChromaDB / FAISS** (Banco de dados vetorial)
- **OpenAI API** (LLM para geração de respostas)


