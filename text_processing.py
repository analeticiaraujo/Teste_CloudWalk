# text_processing.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle
import os

def split_documents_into_chunks(documents):
    """
    Splits a list of LangChain Document objects into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # Tente um tamanho de chunk menor
        chunk_overlap=100,   # E um overlap menor para chunks mais concisos
        length_function=len,
        is_separator_regex=False,
    )

    print(f"Splitting {len(documents)} documents into chunks...")
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    if chunks:
        print("\n--- Sample of first few chunks ---")
        for i, chunk in enumerate(chunks[:3]):
            print(f"Chunk {i+1} (length {len(chunk.page_content)}):")
            print(chunk.page_content[:200] + "...")
            print("-" * 30)
        print("----------------------------------\n")

    return chunks

if __name__ == "__main__":
    docs_pkl_path = "cloudwalk_documents.pkl"
    cloudwalk_docs = []

    if os.path.exists(docs_pkl_path):
        print(f"Loading documents from {docs_pkl_path}...")
        with open(docs_pkl_path, "rb") as f:
            cloudwalk_docs = pickle.load(f)
        print(f"Loaded {len(cloudwalk_docs)} documents.")
    else:
        print(f"'{docs_pkl_path}' not found. Please run your data ingestion scripts and combine them first.")
        exit("Please run data_ingestion_cloudwalk.py, data_ingestion_infinitepay.py, data_ingestion_reclameaqui.py, and combine_all_scraped_documents.py first.")

    if cloudwalk_docs:
        chunks = split_documents_into_chunks(cloudwalk_docs)
        with open("cloudwalk_chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
        print("Chunking complete. Proceed to embedding and vector store creation.")
    else:
        print("No documents to chunk. Exiting.")