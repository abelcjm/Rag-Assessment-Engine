import os
import re
import gc
import torch
from dotenv import load_dotenv
from supabase.client import create_client, Client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

load_dotenv()
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("CRITICAL: Supabase URL or Key missing from environment.")

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"trust_remote_code": True, "device": DEVICE},
    encode_kwargs={"normalize_embeddings": True}
)

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
vector_db = SupabaseVectorStore(
    client=supabase_client,
    embedding=embedding_model,
    table_name="Documents",
    query_name="match_documents"
)

# Cross-Encoder using model_kwargs to resolve the deprecation warning
reranker_model = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    device=DEVICE,
    max_length=512,
    model_kwargs={"torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32}
)

def _normalize_tokens(text: str) -> set:
    """Extract lowercase alphanumeric tokens, ignoring underscores, commas, and punctuation."""
    return set(re.findall(r'[a-zA-Z0-9]+', text.lower()))

def retrieve_and_rerank(query: str, top_k: int = 40, final_k: int = 7, reference_filter: list = None):
    """
    Two-stage retrieval pipeline:
    1. Vector similarity search against Supabase pgvector.
    2. Resilient token-overlap filtering for sanitized/prefixed textbook filenames.
    3. Batched FP16 Cross-Encoder reranking on RTX GPU.
    """
    if reference_filter is None:
        reference_filter = ["All"]

    # Fetch initial pool of candidates from pgvector
    initial_results = vector_db.similarity_search(query, k=top_k)

    if not initial_results:
        return []

    # In-memory filter with normalized token matching
    if "All" not in reference_filter:
        filtered = []
        for doc in initial_results:
            source_raw = doc.metadata.get("source", "")
            source_tokens = _normalize_tokens(source_raw)

            matched = False
            for ref in reference_filter:
                ref_tokens = _normalize_tokens(ref)
                # Match if at least 3 key words from the target title are found in the source string
                if ref_tokens and len(ref_tokens.intersection(source_tokens)) >= min(3, len(ref_tokens)):
                    matched = True
                    break

            if matched:
                filtered.append(doc)
        initial_results = filtered

    if not initial_results:
        return []

    # Batched FP16 Cross-Encoder Reranking
    pairs = [[query, doc.page_content] for doc in initial_results]

    with torch.inference_mode():
        scores = reranker_model.predict(
            pairs,
            batch_size=8 if DEVICE == "cuda" else 4,
            show_progress_bar=False
        )

    scored_results = sorted(zip(initial_results, scores), key=lambda x: x[1], reverse=True)
    best_chunks = [doc for doc, _ in scored_results[:final_k]]

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return best_chunks