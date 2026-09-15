import os
import pymupdf 
import tempfile
import gc  
from dotenv import load_dotenv

load_dotenv()

# Disable Torch JIT compilation and Dynamo to avoid C++ runtime compilation errors in constrained environments
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_hybrid_pdf(pdf_path, docling_converter, docling_chunker, text_splitter):
    """
    Extracts and chunks PDF pages using a dual-path heuristic routing strategy.

    Standard text pages bypass heavy vision models to maximize throughput, while complex
    pages (tables, vectors, formulas, low text) route through Docling for structural layout recovery.

    Args:
        pdf_path (str): Filepath to the target PDF.
        docling_converter (DocumentConverter): Configured Docling converter instance.
        docling_chunker (HybridChunker): Layout-aware chunker for Docling documents.
        text_splitter (RecursiveCharacterTextSplitter): Splitter for standard text pages.

    Returns:
        list[Document]: LangChain Document objects containing chunked text and metadata.
    """
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
        
        # Route to Docling if the page has sparse text, vector diagrams, tables, rotation, or dense layouts
        if len(fast_text) < 100 or len(drawing_list) > 15 or len(table_list) > 0 or is_rotated or len(block_list) > 40:
            print(f"  -> Page {page_num + 1}: Complexity detected. Routing to Docling (with LaTeX enrichment)...")
            temp_pdf = pymupdf.open()
            temp_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_path = temp_file.name
                
            temp_pdf.save(temp_path)
            temp_pdf.close()
            
            conversion_result = docling_converter.convert(temp_path)
            
            # Preserve structural boundaries and repeat table headers across split boundaries
            chunk_iterator = docling_chunker.chunk(dl_doc=conversion_result.document)
            for chunk in chunk_iterator:
                chunk_text = docling_chunker.contextualize(chunk)
                extracted_documents.append(
                    Document(
                        page_content=chunk_text,
                        metadata={"source": os.path.basename(pdf_path), "page": page_num + 1}
                    )
                )
            
            # Explicitly free Docling backend C++ memory buffers to prevent memory leaks during large PDF processing
            if hasattr(conversion_result.input, '_backend'):
                conversion_result.input._backend.unload()
            del conversion_result
            gc.collect() 
            os.remove(temp_path)
        else:
            # Unified visual confirmation message
            print(f"  -> Page {page_num + 1}: Standard text extracted via PyMuPDF.")
            
            # Split standard text into discrete chunks to prevent vector semantic dilution in the embedding space
            raw_doc = [Document(
                page_content=fast_text,
                metadata={"source": os.path.basename(pdf_path), "page": page_num + 1}
            )]
            split_chunks = text_splitter.split_documents(raw_doc)
            extracted_documents.extend(split_chunks)
        
    doc.close()
    return extracted_documents

def batch_ingest_hybrid(pdf_directory):
    """
    Ingests all unindexed PDFs in a directory into ChromaDB.

    Checks ChromaDB metadata to prevent redundant processing of existing source files.

    Args:
        pdf_directory (str): Directory path containing source PDF textbooks.
    """
    print(f"Step 1: Scanning '{pdf_directory}' for academic literature...")
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No PDFs found. Exiting.")
        return

    print("Checking database for already ingested files...")
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5", 
        model_kwargs={'trust_remote_code': True, 'device': 'cuda'} 
    )
    
    vector_db = Chroma(
        persist_directory="./chroma_db_store", 
        embedding_function=embedding_model
    )
    
    # Query vector store metadata to avoid re-embedding previously processed documents
    existing_data = vector_db.get(include=['metadatas'])
    existing_sources = set([meta['source'] for meta in existing_data['metadatas'] if 'source' in meta]) if existing_data and existing_data['metadatas'] else set()
                
    new_pdf_files = [f for f in pdf_files if f not in existing_sources]
    
    if not new_pdf_files:
        print("All files have already been ingested. Nothing new to process.")
        return
        
    print(f"Found {len(new_pdf_files)} new file(s) to ingest. Skipping {len(existing_sources)} existing file(s).")
    
    # Enable formula enrichment to convert math and psychometric scoring equations into LaTeX
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_formula_enrichment = True

    docling_converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    docling_chunker = HybridChunker(max_tokens=512, merge_peers=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    all_raw_documents = []

    for file_name in new_pdf_files:
        file_path = os.path.join(pdf_directory, file_name)
        raw_docs = process_hybrid_pdf(file_path, docling_converter, docling_chunker, text_splitter)
        all_raw_documents.extend(raw_docs)

    print("\nStep 2: Appending chunks to the local vector database in batches...")
    
    # Insert in bounded batches to maximize GPU tensor parallelization without causing OOM errors
    batch_size = 100 
    total_chunks = len(all_raw_documents)
    
    for i in range(0, total_chunks, batch_size):
        batch = all_raw_documents[i:i + batch_size]
        vector_db.add_documents(batch)
        print(f"  -> Progress: Embedded and stored chunks {i} to {min(i + batch_size, total_chunks)} of {total_chunks}")
        gc.collect()
    
    print("\n--- Hybrid Ingestion Complete ---")

if __name__ == "__main__":
    target_directory = "./textbooks"
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    else:
        batch_ingest_hybrid(target_directory)