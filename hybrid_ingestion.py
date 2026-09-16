import os
import gc
import pymupdf
import tempfile
import torch
from celery import shared_task
from supabase.client import create_client, Client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from dotenv import load_dotenv

load_dotenv()
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# 1. Global Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("CRITICAL: Supabase URL or Key is missing from environment variables.")

print(f"Connecting to Supabase pgvector at {SUPABASE_URL} for Ingestion Worker...")
embedding_model = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5", 
    model_kwargs={'trust_remote_code': True, 'device': DEVICE} 
)

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
vector_db = SupabaseVectorStore(
    client=supabase_client,
    embedding=embedding_model,
    table_name="Documents",
    query_name="match_documents"
)

def process_hybrid_pdf(pdf_path, docling_converter, docling_chunker, text_splitter):
    doc = pymupdf.open(pdf_path)
    extracted_documents = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        fast_text = page.get_text("text").strip()
        drawing_list = page.get_drawings()
        table_list = page.find_tables().tables
        is_rotated = page.rotation != 0
        block_list = page.get_text("blocks")
        
        if len(fast_text) < 100 or len(drawing_list) > 15 or len(table_list) > 0 or is_rotated or len(block_list) > 40:
            temp_pdf = pymupdf.open()
            temp_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_path = temp_file.name
                
            temp_pdf.save(temp_path)
            temp_pdf.close()
            
            conversion_result = docling_converter.convert(temp_path)
            chunk_iterator = docling_chunker.chunk(dl_doc=conversion_result.document)
            
            for chunk in chunk_iterator:
                chunk_text = docling_chunker.contextualize(chunk)
                extracted_documents.append(
                    Document(
                        page_content=chunk_text,
                        metadata={"source": os.path.basename(pdf_path), "page": page_num + 1}
                    )
                )
            
            if hasattr(conversion_result.input, '_backend'):
                conversion_result.input._backend.unload()
            del conversion_result
            gc.collect() 
            os.remove(temp_path)
        else:
            raw_doc = [Document(
                page_content=fast_text,
                metadata={"source": os.path.basename(pdf_path), "page": page_num + 1}
            )]
            split_chunks = text_splitter.split_documents(raw_doc)
            extracted_documents.extend(split_chunks)
        
    doc.close()
    return extracted_documents

@shared_task(bind=True)
def batch_ingest_task(self, pdf_directory):
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    if not pdf_files:
        return {"status": "No PDFs found in directory."}

    # Query Supabase using the exact case-sensitive table name
    response = supabase_client.table("Documents").select("metadata").execute()
    existing_sources = set([
        row["metadata"].get("source") 
        for row in response.data 
        if "metadata" in row and "source" in row["metadata"]
    ])
                
    new_pdf_files = [f for f in pdf_files if f not in existing_sources]
    
    if not new_pdf_files:
        return {"status": "All files already ingested.", "skipped": len(existing_sources)}

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_formula_enrichment = True
    docling_converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    docling_chunker = HybridChunker(max_tokens=512, merge_peers=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    all_raw_documents = []

    for idx, file_name in enumerate(new_pdf_files):
        self.update_state(state='PROCESSING', meta={'current_file': file_name, 'progress': f"{idx + 1}/{len(new_pdf_files)}"})
        
        file_path = os.path.join(pdf_directory, file_name)
        raw_docs = process_hybrid_pdf(file_path, docling_converter, docling_chunker, text_splitter)
        all_raw_documents.extend(raw_docs)

    batch_size = 100 
    total_chunks = len(all_raw_documents)
    
    for i in range(0, total_chunks, batch_size):
        batch = all_raw_documents[i:i + batch_size]
        vector_db.add_documents(batch)
        gc.collect()

    return {
        "status": "Success", 
        "files_processed": len(new_pdf_files), 
        "chunks_embedded": total_chunks
    }