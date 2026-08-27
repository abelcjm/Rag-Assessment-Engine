import os
import pymupdf 
import tempfile
import gc  
from dotenv import load_dotenv

# Load the .env file BEFORE importing docling or langchain
load_dotenv()
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def process_hybrid_pdf(pdf_path, docling_converter):
    print(f"\nAnalyzing Document: {os.path.basename(pdf_path)}")
    doc = pymupdf.open(pdf_path)
    extracted_documents = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        fast_text = page.get_text("text").strip()
        drawing_list = page.get_drawings()
        table_list = page.find_tables().tables
        is_rotated = page.rotation != 0
        block_list = page.get_text("blocks")
        
        page_content = ""
        
        if len(fast_text) < 100 or len(drawing_list) > 15 or len(table_list) > 0 or is_rotated or len(block_list) > 40:
            print(f"  -> Page {page_num + 1}: Complexity detected. Routing to Docling...")
            temp_pdf = pymupdf.open()
            temp_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_path = temp_file.name
                
            temp_pdf.save(temp_path)
            temp_pdf.close()
            
            conversion_result = docling_converter.convert(temp_path)
            page_content = conversion_result.document.export_to_markdown()
            
            # --- CRITICAL MEMORY LEAK FIX ---
            if hasattr(conversion_result.input, '_backend'):
                conversion_result.input._backend.unload()
            del conversion_result
            gc.collect() 
            # --------------------------------
            
            os.remove(temp_path)
        else:
            print(f"  -> Page {page_num + 1}: Standard text detected. Routing to Fast Extractor...")
            page_content = fast_text
            
        extracted_documents.append(
            Document(
                page_content=page_content,
                metadata={"source": os.path.basename(pdf_path), "page": page_num + 1}
            )
        )
        
    doc.close()
    return extracted_documents

def batch_ingest_hybrid(pdf_directory):
    print(f"Step 1: Scanning '{pdf_directory}' for academic literature...")
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No PDFs found. Exiting.")
        return

    print("Checking database for already ingested files...")
    
    # Restored to HuggingFace and pointing to the GPU
    embedding_model = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5", 
        model_kwargs={'trust_remote_code': True, 'device': 'cuda'} 
    )
    
    vector_db = Chroma(
        persist_directory="./chroma_db_store", 
        embedding_function=embedding_model
    )
    
    existing_data = vector_db.get(include=['metadatas'])
    existing_sources = set()
    
    if existing_data and existing_data['metadatas']:
        for meta in existing_data['metadatas']:
            if 'source' in meta:
                existing_sources.add(meta['source'])
                
    new_pdf_files = [f for f in pdf_files if f not in existing_sources]
    
    if not new_pdf_files:
        print("All files have already been ingested. Nothing new to process.")
        return
        
    print(f"Found {len(new_pdf_files)} new file(s) to ingest. Skipping {len(existing_sources)} existing file(s).")
    
    docling_converter = DocumentConverter()
    all_raw_documents = []

    for file_name in new_pdf_files:
        file_path = os.path.join(pdf_directory, file_name)
        raw_docs = process_hybrid_pdf(file_path, docling_converter)
        all_raw_documents.extend(raw_docs)

    print(f"\nStep 2: Chunking the extracted data...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    final_chunks = text_splitter.split_documents(all_raw_documents)

    print("\nStep 3: Appending chunks to the local vector database in batches...")
    
    # Increased batch size to maximize GPU efficiency
    batch_size = 100 
    total_chunks = len(final_chunks)
    
    for i in range(0, total_chunks, batch_size):
        batch = final_chunks[i:i + batch_size]
        vector_db.add_documents(batch)
        print(f"  -> Progress: Embedded and stored chunks {i} to {min(i + batch_size, total_chunks)} of {total_chunks}")
        gc.collect()
    
    print("\n--- Hybrid Ingestion Complete ---")

if __name__ == "__main__":
    target_directory = "./textbooks"
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        print(f"Created directory '{target_directory}'. Please place your PDFs inside and run again.")
    else:
        batch_ingest_hybrid(target_directory)