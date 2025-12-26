import os
import tempfile
from decouple import config
import streamlit as st
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import  ChatOpenAI, OpenAIEmbeddings


os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')

persisted_directory = 'db'

def proccess_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    os.remove(temp_file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
    )
    chunks = text_splitter.split_documents(documents=docs)
    return chunks


def load_existing_vector_store():
    if os.path.exists(os.path.join(persisted_directory)):
        vector_store = Chroma(
            persist_directory=persisted_directory,
            embedding_function=OpenAIEmbeddings()
        )
        return vector_store
    return None

def add_vector_store(chunks, vector_stored=None):
    if vector_stored:
        vector_stored.add_documents(chunks)
    else:
        vector_stored = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory=persisted_directory
        )
    return vector_stored


def ask_question(model, query, vector_stores):
    llm = ChatOpenAI(model=model)
    retriever = vector_stores.as_retriever()

    system_prompt = """
    Use o contexto para responder as perguntas.
    Se não encontrar uma resposta no contexto,
    explique que não há informações disponíveis.
    Responda em formato de markdown e com visualizações elaboradas e interativas.
    Contexto: {context}
    """
    messages = [('system', system_prompt,)]
    for message in st.session_state.messages:
        messages.append((message.get('role'), message.get('content')))
    messages.append(('human', '{input}'))

    prompt = ChatPromptTemplate.from_messages(messages)

    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain

    )
    response = chain.invoke({'input': query})
    return response.get('answer')

vector_store = load_existing_vector_store()

st.set_page_config(
    page_title='AI assistant',
    page_icon='👾'
)

st.header('The best AI assistant 🤖')


with st.sidebar:
    st.header('AI can help you with the files you need ! ')
    files_uploaded = st.file_uploader(
        label= 'Upload  a file!',
        type=['pdf'],
        accept_multiple_files=True,
    )

    if files_uploaded:
        with st.spinner('Processing documents...'):
            all_chunks = []
            for uploaded_file in files_uploaded:
                chunks = proccess_pdf(file=uploaded_file)
                all_chunks.extend(chunks)
                vector_store = add_vector_store(all_chunks)

    model_options = [
        'gpt-3.5-turbo',
        'gpt-4',
        'gpt-4-turbo',
        'gpt-4o-mini',
        'gpt-4o',
    ]

selected_model = st.sidebar.selectbox(
    label='Model',
    options=model_options
)

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

question = st.chat_input('Como posso ajudar?')

if vector_store and question:
    with st.spinner('Searching information...'):
        for message in st.session_state.messages:
            st.chat_message(message.get('role')).write(message.get('content'))

        st.chat_message('user').write(question)
        st.session_state.messages.append({'role': 'user', 'content': question})

        response = ask_question(
            model=selected_model,
            query=question,
            vector_stores=vector_store,
        )

        st.chat_message('ai').write(response)
        st.session_state.messages.append({'role': 'ai', 'content': response})