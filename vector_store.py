import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pickle 

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Função para inicializar o modelo de embeddings do Gemini
def get_gemini_embeddings_model():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Função para criar o vector store com os chunks de texto
# Adiciona um parâmetro embeddings_model para que a função seja mais flexível
def create_vector_store(chunks, persist_directory="./chroma_db", embeddings_model=None):
    """
    Generates embeddings for text chunks and stores them in a ChromaDB vector store.
    Utilizes the specified embeddings_model or defaults to Gemini's embedding.
    """
    if embeddings_model is None:
        print("Initializing default embedding model (Gemini's embedding-001)...")
        embeddings = get_gemini_embeddings_model()
    else:
        print("Using provided embedding model...")
        embeddings = embeddings_model

    print(f"Creating ChromaDB vector store in {persist_directory}...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("Vector store created and persisted.")
    return vector_store

# Adicionado um parâmetro embeddings_model para que a função seja mais flexível
def load_vector_store(persist_directory="./chroma_db", embeddings_model=None):
    """
    Loads an existing ChromaDB vector store.
    Utilizes the specified embeddings_model or defaults to Gemini's embedding.
    """
    if embeddings_model is None:
        print("Initializing default embedding model (Gemini's embedding-001) for loading...")
        embeddings = get_gemini_embeddings_model()
    else:
        print("Using provided embedding model for loading...")
        embeddings = embeddings_model

    print(f"Loading ChromaDB vector store from {persist_directory}...")
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    print("Vector store loaded.")
    return vector_store

if __name__ == "__main__":
    db_path = "./chroma_db"
    
    # Inicializa o modelo de embeddings AQUI para usar tanto na criação quanto no carregamento
    current_embeddings_model = get_gemini_embeddings_model()

    if not os.path.exists(db_path) or not any(File.endswith(".bin") or File.endswith(".parquet") or File.endswith(".sqlite3") for File in os.listdir(db_path)):
        print("Vector store does not exist. Attempting to create it now...")
        chunks_pkl_path = "cloudwalk_chunks.pkl"
        chunks = []
        if os.path.exists(chunks_pkl_path):
            print(f"Loading chunks from {chunks_pkl_path}...")
            with open(chunks_pkl_path, "rb") as f:
                chunks = pickle.load(f)
            print(f"Loaded {len(chunks)} chunks.")
        else:
            print(f"'{chunks_pkl_path}' not found. You need to run 'data_ingestion.py' and 'text_processing.py' first.")
            exit("Please run `python data_ingestion.py` and `python text_processing.py` first to generate chunks.")

        if chunks:
            vector_store = create_vector_store(chunks, persist_directory=db_path, embeddings_model=current_embeddings_model)
        else:
            print("No chunks found to create vector store. Exiting.")
            exit()
    else:
        print("Vector store already exists. Loading it...")
        vector_store = load_vector_store(persist_directory=db_path, embeddings_model=current_embeddings_model)

    # Test the retriever
    print("\n--- Testing vector store retrieval ---")
    query = "What are CloudWalk's main products?"
    retriever = vector_store.as_retriever()
    retrieved_docs = retriever.invoke(query)
    print(f"Retrieved {len(retrieved_docs)} documents for query: '{query}'")
    if retrieved_docs:
        print("First retrieved document content snippet:")
        print(retrieved_docs[0].page_content[:300] + "...")
    print("--------------------------------------\n")