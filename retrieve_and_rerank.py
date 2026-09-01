import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

def retrieve_and_rerank(query, top_k=12, final_k=5, reference_filter=["All"]):
    print(f"Step 1: Embedding query and retrieving initial {top_k} candidate chunks...")
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5", 
        model_kwargs={'trust_remote_code': True, 'device': 'cpu'} 
    )
    
    vector_db = Chroma(
        persist_directory="./chroma_db_store", 
        embedding_function=embedding_model
    )
    
    search_filter = None
    if "All" not in reference_filter:
        search_filter = {"source": {"$in": reference_filter}}
        print(f"        Applying metadata filter: {search_filter}")
        
    if search_filter:
        initial_results = vector_db.similarity_search(query, k=top_k, filter=search_filter)
    else:
        initial_results = vector_db.similarity_search(query, k=top_k)
    
    print(f"\nStep 2: Reranking {len(initial_results)} candidates down to top {final_k} chunks...")
    if not initial_results:
        print("\n--- Retrieval Warning: No chunks found matching query/filter. ---")
        return []
        
    reranker_model = CrossEncoder('BAAI/bge-reranker-v2-m3')
    pairs = [[query, doc.page_content] for doc in initial_results]
    scores = reranker_model.predict(pairs)
    
    scored_results = list(zip(initial_results, scores))
    scored_results.sort(key=lambda x: x[1], reverse=True)
    
    best_chunks = [doc for doc, score in scored_results[:final_k]]
    
    print(f"Successfully selected top {len(best_chunks)} dense chunks for generation.")
    return best_chunks

if __name__ == "__main__":
    test_query = "Abnormal Psychology Explain the cognitive theories in the development of psychological disorders clinical diagnostic criteria symptoms"
    best_context = retrieve_and_rerank(test_query, top_k=12, final_k=5)
    print(f"\nRetrieved {len(best_context)} chunks.")