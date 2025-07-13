This environment contains the scripts and dependencies to prepare data, generate embeddings, and run the RAG chatbot using Streamlit.

Script Structure data_ingestion.py Collects data from CloudWalk websites and saves the processed documents.

text_processing.py Splits the documents into chunks to facilitate vectorization.

vector_store.py Generates embeddings from the chunks and creates the ChromaDB vector database.

chatbot_app.py Streamlit interface for interacting with the chatbot, utilizing Gemini and semantic search.

How to Prepare the Environment Install Dependencies Activate your virtual environment and run:

Bash

pip install -r requirements.txt requirements.txt (Suggested content for your requirements.txt file):

python-dotenv requests beautifulsoup4 langchain-community langchain-text-splitters langchain-google-genai langchain-chroma streamlit numpy==1.26.4 # Specific version to avoid NumPy 2.x compatibility issues torch==2.2.0 # Adjust based on your system (CPU or CUDA) if you face issues sentence-transformers # Required for some underlying functionalities, or if using HuggingFaceEmbeddings Note: For torch, if you have an NVIDIA GPU and CUDA installed, please refer to the official PyTorch website (pytorch.org/get-started/locally/) for the specific installation command that leverages your GPU.

Prepare the Data Execute the scripts in order:

Bash

python data_ingestion.py python text_processing.py python vector_store.py Tip: If you've previously run data_ingestion.py and text_processing.py, and want to re-create the vector store, you can delete the ./chroma_db folder before running vector_store.py.

Run the Chatbot Bash

streamlit run chatbot_app.py Notes The vector database is saved in ./chroma_db.

If you encounter the error Vector store not found (or chroma_db folder is missing/empty), execute the data preparation scripts again.

Add your Gemini API key to the .env file:

GOOGLE_API_KEY=your_token_here USER_AGENT=CloudWalkBot/1.0 Requirements Python 3.10+

The packages listed in the requirements.txt section above.

Questions or Issues? Check the script logs and ensure all intermediate files (cloudwalk_documents.pkl, cloudwalk_chunks.pkl, etc.) are present and correct in your project's root directory.
