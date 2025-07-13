from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle
import os

# Esse script é responsável por dividir documentos em pedaços menores
# para facilitar o processamento e a indexação em um banco de dados vetorial.
def split_documents_into_chunks(documents):
    """
    Splits a list of LangChain Document objects into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    print(f"Splitting {len(documents)} documents into chunks...")
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # Mostrar alguns exemplos dos chunks criados
    if chunks:
        print("\n--- Sample of first few chunks ---")
        for i, chunk in enumerate(chunks[:3]):
            print(f"Chunk {i+1} (length {len(chunk.page_content)}):")
            print(chunk.page_content[:200] + "...")
            print("-" * 30)
        print("----------------------------------\n")

    return chunks

# Executa o script se for chamado diretamente
if __name__ == "__main__":
    # Se o arquivo de documentos já existir, carrega os documentos dele
    # Caso contrário, executa a ingestão de dados para criar o arquivo pkl
    docs_pkl_path = "cloudwalk_documents.pkl"
    cloudwalk_docs = []

    if os.path.exists(docs_pkl_path):
        print(f"Loading documents from {docs_pkl_path}...")
        with open(docs_pkl_path, "rb") as f:
            cloudwalk_docs = pickle.load(f)
        print(f"Loaded {len(cloudwalk_docs)} documents.")
    else:
        print(f"'{docs_pkl_path}' not found. Running data ingestion...")
        # Só executa a ingestão de dados se o arquivo pkl não existir
        from data_ingestion import load_cloudwalk_data
        cloudwalk_docs = load_cloudwalk_data()
        # O load_cloudwalk_data já salva os documentos no pkl, portanto não é necessário salvar novamente

    # Se houver documentos carregados, divide-os em pedaços menores
    if cloudwalk_docs:
        chunks = split_documents_into_chunks(cloudwalk_docs)
        with open("cloudwalk_chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
        print("Chunking complete. Proceed to embedding and vector store creation.")
    else:
        print("No documents to chunk. Exiting.")