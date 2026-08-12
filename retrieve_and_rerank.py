import os
# Bypass the PyTorch compiler check on Windows 
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

def retrieve_and_rerank(query, top_k=10, final_k=3):
    print(f"Step 1: Embedding the query and retrieving initial {top_k} chunks...")
    
    # 1. Initialize the Nomic model used for the database storage
    embedding_model = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5", 
        model_kwargs={'trust_remote_code': True, 'device': 'cpu'}
    )
    
    # 2. Connect to the local Chroma database
    vector_db = Chroma(
        persist_directory="./chroma_db_store", 
        embedding_function=embedding_model
    )
    
    # 3. Perform the initial Cosine Similarity search
    initial_results = vector_db.similarity_search(query, k=top_k)
    
    print("\nStep 2: Reranking results to eliminate semantic noise...")
    
    # 4. Initialize the specialized BGE Reranker
    reranker_model = CrossEncoder('BAAI/bge-reranker-v2-m3')
    
    # 5. Format the data into pairs (Query, Chunk Text) for the cross-encoder
    pairs = [[query, doc.page_content] for doc in initial_results]
    
    # 6. Calculate the strict relevance scores
    scores = reranker_model.predict(pairs)
    
    # 7. Sort the initial chunks based on the reranker's new scores
    scored_results = list(zip(initial_results, scores))
    scored_results.sort(key=lambda x: x[1], reverse=True)
    
    # 8. Extract the absolute best, highly concentrated chunks
    best_chunks = [doc for doc, score in scored_results[:final_k]]
    
    print(f"\n--- Retrieval Complete ---")
    print(f"Successfully filtered the top {final_k} most relevant chunks for generation.")
    return best_chunks

if __name__ == "__main__":
    # A test query targeting specific competencies from the TOS
    test_query = "What are the cognitive theories in the development of psychological disorders?"
    best_context = retrieve_and_rerank(test_query)
    
    print("\n--- Top Retrieved Context ---")
    print(best_context[0].page_content)
    print("\nMetadata:", best_context[0].metadata)