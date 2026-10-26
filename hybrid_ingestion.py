import os
import gc
import time
from io import BytesIO

import pymupdf
import torch
from dotenv import load_dotenv
from celery import shared_task
from supabase.client import create_client, Client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

load_dotenv()
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("CRITICAL: Supabase URL or Key is missing from environment variables.")

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5", 
    model_kwargs={'trust_remote_code': True, 'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
vector_db = SupabaseVectorStore(
    client=supabase_client,
    embedding=embedding_model,
    table_name="Documents",
    query_name="match_documents"
)

def process_hybrid_pdf(pdf_bytes, file_name, docling_converter, docling_chunker, text_splitter, indexed_path, tags):
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    extracted_documents = []
    total_pages = len(doc)
    print(f"[*] Starting FAST-MODE processing for {file_name} ({total_pages} pages)...")

    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        fast_text = page.get_text("text").strip()
        table_list = page.find_tables().tables
        
        chunk_metadata = {
            "source": file_name,
            "page": page_num + 1,
            "storage_path": indexed_path,
            "tags": tags
        }
        
        needs_docling = len(fast_text) < 50 or len(table_list) > 0
        
        if needs_docling:
            temp_pdf = pymupdf.open()
            temp_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
            page_bytes = temp_pdf.tobytes()
            temp_pdf.close()
            
            buf = BytesIO(page_bytes)
            source = DocumentStream(name=f"page_{page_num}.pdf", stream=buf)
            
            try:
                conversion_result = docling_converter.convert(source)
                chunk_iterator = docling_chunker.chunk(dl_doc=conversion_result.document)
                
                for chunk in chunk_iterator:
                    chunk_text = docling_chunker.contextualize(chunk)
                    extracted_documents.append(
                        Document(page_content=chunk_text, metadata=chunk_metadata)
                    )
                
                if hasattr(conversion_result.input, '_backend'):
                    conversion_result.input._backend.unload()
                del conversion_result
            except Exception as e:
                print(f"[Docling fallback to PyMuPDF on page {page_num+1}]: {e}")
                raw_doc = [Document(page_content=fast_text, metadata=chunk_metadata)]
                extracted_documents.extend(text_splitter.split_documents(raw_doc))
        else:
            raw_doc = [Document(page_content=fast_text, metadata=chunk_metadata)]
            split_chunks = text_splitter.split_documents(raw_doc)
            extracted_documents.extend(split_chunks)
        
        if (page_num + 1) % 50 == 0:
            print(f"[*] Progress: {page_num + 1}/{total_pages} pages processed.")

    doc.close()
    print(f"[✓] Extraction finished. Generated {len(extracted_documents)} total chunks.")
    return extracted_documents

@shared_task(bind=True)
def batch_ingest_task(self, request_id, storage_path, tags):
    if not storage_path or not request_id:
        raise ValueError("Missing request_id or storage_path.")

    file_name = os.path.basename(storage_path)
    indexed_path = storage_path.replace("pending/", "indexed/")

    self.update_state(state='PROCESSING', meta={'step': 'Checking Database', 'details': f'Verifying {file_name} status...'})
    response = supabase_client.table("Documents") \
        .select("id") \
        .contains("metadata", {"source": file_name}) \
        .limit(1) \
        .execute()
                
    if response.data:
        try:
            if storage_path.startswith("pending/"):
                supabase_client.storage.from_("textbooks").move(storage_path, indexed_path)
        except Exception:
            pass

        supabase_client.table("Material Requests").update({
            "file_path": indexed_path,
            "status": "Approved & Indexed"
        }).eq("id", request_id).execute()

        return {"status": "Success", "message": f"{file_name} is already indexed. Status verified."}

    self.update_state(state='PROCESSING', meta={'step': 'Downloading PDF', 'details': f'Fetching {file_name} from pending storage...'})

    max_retries = 3
    pdf_bytes = None
    
    for attempt in range(max_retries):
        try:
            pdf_bytes = supabase_client.storage.from_("textbooks").download(storage_path)
            break
        except Exception as e:
            if attempt == max_retries - 1:
                supabase_client.table("Material Requests").update({
                    "status": "Rejected",
                    "reject_reason": f"Storage download failed: {str(e)}"
                }).eq("id", request_id).execute()
                raise RuntimeError(f"Download failed after {max_retries} attempts: {e}")
            time.sleep(2 ** attempt)

    self.update_state(state='PROCESSING', meta={'step': 'Extracting Layout', 'details': f'Parsing complex layouts in {file_name}...'})

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_formula_enrichment = False
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4,
        device=AcceleratorDevice.CPU
    )

    docling_converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    
    docling_chunker = HybridChunker(max_tokens=1000, merge_peers=True)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "(?<=\. )", " "],
        chunk_size=1200, 
        chunk_overlap=200
    )
    
    try:
        raw_docs = process_hybrid_pdf(pdf_bytes, file_name, docling_converter, docling_chunker, text_splitter, indexed_path, tags)

        self.update_state(state='PROCESSING', meta={'step': 'Embedding', 'details': f'Pushing {len(raw_docs)} chunks to vector database...'})
        
        batch_size = 100 
        total_chunks = len(raw_docs)
        
        for i in range(0, total_chunks, batch_size):
            batch = raw_docs[i:i + batch_size]
            vector_db.add_documents(batch)
            gc.collect()

        self.update_state(state='PROCESSING', meta={'step': 'Finalizing', 'details': f'Updating storage and approval status for {file_name}...'})
        
        try:
            if storage_path.startswith("pending/"):
                supabase_client.storage.from_("textbooks").move(storage_path, indexed_path)
        except Exception as move_err:
            print(f"[Storage Move Warning]: {move_err}")

        update_res = supabase_client.table("Material Requests").update({
            "file_path": indexed_path,
            "status": "Approved & Indexed"
        }).eq("id", request_id).execute()

        print(f"[INGESTION COMPLETE]: {file_name} updated to Approved & Indexed: {update_res.data}")

    except Exception as e:
        try:
            supabase_client.table("Documents").delete().eq("metadata->>source", file_name).execute()
        except Exception:
            pass

        supabase_client.table("Material Requests").update({
            "status": "Rejected",
            "reject_reason": f"Processing crashed: {str(e)}"
        }).eq("id", request_id).execute()

        raise RuntimeError(f"Processing crashed and rolled back: {str(e)}")

    return {
        "status": "Success", 
        "file_processed": file_name, 
        "chunks_embedded": total_chunks,
        "new_storage_path": indexed_path
    }