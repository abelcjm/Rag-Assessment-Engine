import os
# Bypass the PyTorch compiler check on Windows again
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from chunk_data import chunk_tos_markdown
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def build_local_vector_database():
    print("Step 1: Extracting and chunking the Markdown document...")
    # This calls the function from your previous script to get the 85 chunks
    chunks = chunk_tos_markdown("Parsed_PRC_TOS.md")
    
    print("\nStep 2: Initializing the Nomic Embedding Model...")
    # Downloading and loading the embedding model into local memory
    # We use trust_remote_code=True because Nomic models require custom architecture files
    embedding_model = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5", 
        model_kwargs={'trust_remote_code': True, 'device': 'cpu'}
    )

    print("\nStep 3: Depositing embeddings into Local ChromaDB...")
    # This processes all 85 chunks, converts them to vectors, 
    # and saves them into a new folder named 'chroma_db_store'
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model, 
        persist_directory="./chroma_db_store"
    )
    
    print("\n--- Vector Database Successfully Built! ---")
    print("Your non-parametric external memory is now active and stored locally.")

if __name__ == "__main__":
    build_local_vector_database()