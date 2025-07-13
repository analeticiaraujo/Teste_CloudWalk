import os
import streamlit as st
import pickle
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings # For LLM and embeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage

from vector_store import load_vector_store, create_vector_store
from data_ingestion import load_cloudwalk_data
from text_processing import split_documents_into_chunks # Make sure this function exists and works

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# --- Configuration ---
PERSIST_DIRECTORY = "./chroma_db"
LLM_MODEL_NAME = "gemini-2.5-flash" # Atualizado para Gemini 2.5 Flash

# inicializa o Streamlit app
# Isso evita recarregar o vetor e o LLM em cada execução do Streamlit
@st.cache_resource # Cache o recurso para evitar recarregamento desnecessário
def get_vector_store_and_llm():
    llm = None
    try:
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
            st.stop()
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0.2)
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        st.info("Please ensure 'langchain-google-genai' is installed and your API key is valid.")
        st.stop()

    db_path = PERSIST_DIRECTORY
    vector_store = None
    # Usar o modelo de embedding compatível com Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # verifica se o diretório do banco de dados existe e contém arquivos relevantes
    if not os.path.exists(db_path) or not any(File.endswith(".bin") or File.endswith(".parquet") or File.endswith(".sqlite3") for File in os.listdir(db_path)):
        st.info("Vector store not found or empty. Attempting to create it now...")
        
        # 1. Tenta carregar documentos do pkl
        # Se não existir, faz a raspagem dos dados
        docs_pkl_path = "cloudwalk_documents.pkl"
        cloudwalk_docs = []
        if os.path.exists(docs_pkl_path):
            st.info(f"Loading documents from {docs_pkl_path}...")
            with open(docs_pkl_path, "rb") as f:
                cloudwalk_docs = pickle.load(f)
            st.success(f"Loaded {len(cloudwalk_docs)} documents from {docs_pkl_path}.")
        else:
            st.warning("`cloudwalk_documents.pkl` not found. Running data ingestion (web scraping). This may take a moment.")
            cloudwalk_docs = load_cloudwalk_data()
            if not cloudwalk_docs:
                st.error("No documents loaded from web scraping. Please check URLs or internet connection.")
                st.stop()
            # O load_cloudwalk_data já salva o pkl, então não é necessário salvar novamente aqui

        # 2. Torna os documentos em pedaços (chunks)
        st.info("Splitting documents into chunks...")
        chunks = split_documents_into_chunks(cloudwalk_docs)
        if not chunks:
            st.error("No chunks created from documents. Check document content or chunking parameters.")
            st.stop()
        
        # 3. Cria o vetor de armazenamento
        st.info("Creating and persisting vector store...")
        try:
            # Passa os embeddings do Gemini para create_vector_store
            vector_store = create_vector_store(chunks, persist_directory=db_path, embeddings_model=embeddings)
            st.success("Vector store created and saved!")
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            st.info("Please ensure 'sentence-transformers' (if used) and 'torch' are correctly installed, or check Google API Key for embeddings.")
            st.stop()
    else:
        st.info("Loading existing vector store...")
        try:
            # Passa os embeddings do Gemini para load_vector_store
            vector_store = load_vector_store(persist_directory=db_path, embeddings_model=embeddings)
            st.success("Vector store loaded!")
        except Exception as e:
            st.error(f"Error loading vector store: {e}")
            st.info("Consider deleting the `./chroma_db` folder and running `python vector_store.py` again to recreate it.")
            st.stop()
            
    return vector_store, llm

def setup_rag_chain(vector_store, llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use the following context to answer the user's question:\n\n"
                "{context}"
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    rag_chain = create_retrieval_chain(retriever, document_chain)

    return rag_chain

# Lógica principal do Streamlit
st.set_page_config(page_title="CloudWalk AI Assistant", page_icon=":robot_face:")
st.title("CloudWalk AI Assistant")

# Se não houver histórico de chat, inicializa uma lista vazia
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Mostrar o histórico de chat
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)

# Pega o vetor de armazenamento e o LLM
my_vector_store, my_llm = get_vector_store_and_llm()

# Inicializa o RAG chain, passando o vetor de armazenamento e o LLM
rag_chain = setup_rag_chain(my_vector_store, my_llm)

# Lida com a entrada do usuário
user_query = st.chat_input("Ask me about CloudWalk or InfinitePay...")

# Se o usuário enviar uma pergunta, adiciona ao histórico e processa a resposta
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.spinner("Thinking..."):
        try:
            response = rag_chain.invoke({
                "input": user_query,
                "chat_history": st.session_state.chat_history
            })
            ai_response_content = response["answer"]

            with st.chat_message("AI"):
                st.markdown(ai_response_content)
            st.session_state.chat_history.append(AIMessage(content=ai_response_content))

        # Se ocorrer um erro, exibe uma mensagem de erro
        except Exception as e:
            st.error(f"An error occurred during RAG chain invocation: {e}")
            st.info("Please ensure your Google API key is valid and the LLM is accessible.")