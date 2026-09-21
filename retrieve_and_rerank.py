import os
import torch
import gc
from supabase.client import create_client, Client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# 1. Global Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("CRITICAL: Supabase URL or Key is missing from environment variables.")

# 2. Global Initialization
print(f"Loading embedding model on {DEVICE}...")

# FIXED: Aligned with BAAI models and normalization for accurate cosine similarity
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5", 
    model_kwargs={'trust_remote_code': True, 'device': DEVICE},
    encode_kwargs={'normalize_embeddings': True}
)

print(f"Connecting to Supabase pgvector at {SUPABASE_URL}...")
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
vector_db = SupabaseVectorStore(
    client=supabase_client,
    embedding=embedding_model,
    table_name="Documents",
    query_name="match_documents"
)

print(f"Loading cross-encoder reranker on {DEVICE}...")
reranker_model = CrossEncoder('BAAI/bge-reranker-v2-m3', device=DEVICE)

def retrieve_and_rerank(query, top_k=12, final_k=5, reference_filter=["All"]):
    """
    Executes a two-stage retrieval pipeline: dense vector search followed by cross-encoder reranking.
    """
    print(f"Step 1: Retrieving candidates for query: '{query}'")
    
    # LangChain's Supabase implementation uses exact JSONB matching. 
    search_filter = {}
    fetch_k = top_k
    
    if "All" not in reference_filter and len(reference_filter) == 1:
        search_filter = {"source": reference_filter[0]}
        print(f"        Applying DB-level metadata filter: {search_filter}")
    elif "All" not in reference_filter and len(reference_filter) > 1:
        # FIXED: Fetch a much larger pool to ensure enough candidates survive the Python-level filter
        fetch_k = top_k * 4 
        
    initial_results = vector_db.similarity_search(query, k=fetch_k, filter=search_filter)
    
    # Python-level filtering for multiple textbooks
    if "All" not in reference_filter and len(reference_filter) > 1:
        initial_results = [doc for doc in initial_results if doc.metadata.get("source") in reference_filter]
        # Trim back down to top_k after filtering
        initial_results = initial_results[:top_k]
    
    if not initial_results:
        print("\n--- Retrieval Warning: No chunks found matching query/filter. ---")
        return []
        
    print(f"Step 2: Reranking {len(initial_results)} candidates down to top {final_k} chunks...")
    
    pairs = [[query, doc.page_content] for doc in initial_results]
    scores = reranker_model.predict(pairs)
    
    scored_results = list(zip(initial_results, scores))
    scored_results.sort(key=lambda x: x[1], reverse=True)
    
    best_chunks = [doc for doc, score in scored_results[:final_k]]
    
    # FIXED: Clear PyTorch VRAM to prevent Out of Memory crashes during batch generation
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
        
    print(f"Successfully selected top {len(best_chunks)} chunks.")
    return best_chunks