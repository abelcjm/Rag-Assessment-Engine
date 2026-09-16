import os
import torch
from supabase.client import create_client, Client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# 1. Credentials Setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials in your .env file.")

# 2. Initialize Models & Connection
print(f"Loading embedding model on {DEVICE}...")
embedding_model = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5", 
    model_kwargs={'trust_remote_code': True, 'device': DEVICE} 
)

print(f"Connecting to Supabase at {SUPABASE_URL}...")
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
vector_db = SupabaseVectorStore(
    client=supabase_client,
    embedding=embedding_model,
    table_name="Documents",
    query_name="match_documents"
)

# 3. Test Upload (Ingestion)
print("\n--- Test 1: Uploading Vectors ---")
test_docs = [
    Document(
        page_content="Psychometrics is the field of study concerned with the theory and technique of psychological measurement.", 
        metadata={"source": "test_book_1.pdf", "page": 1}
    ),
    Document(
        page_content="The Big Five personality traits are extraversion, agreeableness, openness, conscientiousness, and neuroticism.", 
        metadata={"source": "test_book_2.pdf", "page": 5}
    ),
]
print("Embedding and pushing dummy chunks to Supabase...")
vector_db.add_documents(test_docs)
print("Upload successful!")

# 4. Test Retrieval (Vector Math)
print("\n--- Test 2: Retrieving Vectors ---")
query = "What is psychological measurement?"
print(f"Querying for: '{query}'")
results = vector_db.similarity_search(query, k=1)

if results:
    print("  [✓] SUCCESS: Retrieved Document!")
    print(f"  Content: {results[0].page_content}")
    print(f"  Source: {results[0].metadata.get('source')} (Page {results[0].metadata.get('page')})")
else:
    print("  [X] FAILURE: No documents retrieved.")
    
# 5. Test Metadata Filtering (Database RPC)
print("\n--- Test 3: Metadata Filtering ---")
filter_query = {"source": "test_book_2.pdf"}
print(f"Querying for 'personality' with filter: {filter_query}")
filtered_results = vector_db.similarity_search("personality", k=1, filter=filter_query)

if filtered_results:
    print("  [✓] SUCCESS: Retrieved Filtered Document!")
    print(f"  Content: {filtered_results[0].page_content}")
    print(f"  Source: {filtered_results[0].metadata.get('source')} (Page {filtered_results[0].metadata.get('page')})")
else:
    print("  [X] FAILURE: No documents retrieved with filter.")

print("\n--- All tests completed! ---")