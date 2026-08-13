import os
import pymupdf 
import tempfile
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
# Bypass the PyTorch compiler check on Windows
os.environ["TORCH_COMPILE_DISABLE"] = "1"

def process_hybrid_pdf(pdf_path, docling_converter):
    """Scans a PDF and routes pages to either fast text extraction or Docling."""
    print(f"\nAnalyzing Document: {os.path.basename(pdf_path)}")
    
    # Open the PDF with PyMuPDF (The Scout)
    doc = pymupdf.open(pdf_path)
    extracted_documents = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Detect if the page has images or vector drawings (indicators of tables/charts)
        image_list = page.get_images(full=True)
        drawing_list = page.get_drawings()
        
        page_content = ""
        
        if image_list or drawing_list:
            print(f"  -> Page {page_num + 1}: Complexity detected (Images/Tables). Routing to Docling...")
            
            # Create a temporary PDF containing ONLY this complex page
            temp_pdf = pymupdf.open()
            temp_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
            
            # Generate a temporary file path, but close the Python lock immediately to prevent Windows Permission Errors
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_path = temp_file.name
                
            # Save the PyMuPDF document to the now-unlocked path
            temp_pdf.save(temp_path)
            temp_pdf.close()
            
            # Process the isolated page with Docling to preserve Markdown structure
            conversion_result = docling_converter.convert(temp_path)
            page_content = conversion_result.document.export_to_markdown()
            
            # Clean up the temporary file immediately
            os.remove(temp_path)
            
        else:
            print(f"  -> Page {page_num + 1}: Plain text detected. Routing to Fast Extractor...")
            # Instantly grab the string, bypassing multimodal overhead
            page_content = page.get_text("text")
            
        # Store the extracted content alongside its source metadata
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
        print("No PDFs found in the target directory. Exiting.")
        return

    # Initialize Docling once to save memory during batch processing
    docling_converter = DocumentConverter()
    all_raw_documents = []

    for file_name in pdf_files:
        file_path = os.path.join(pdf_directory, file_name)
        raw_docs = process_hybrid_pdf(file_path, docling_converter)
        all_raw_documents.extend(raw_docs)

    print(f"\nStep 2: Chunking the extracted data...")
    # Use larger chunks to capture full clinical case studies and textbook definitions
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    
    final_chunks = text_splitter.split_documents(all_raw_documents)
    print(f"Successfully divided the textbooks into {len(final_chunks)} semantic chunks.")

    print("\nStep 3: Appending chunks to the local vector database...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5", 
        model_kwargs={'trust_remote_code': True, 'device': 'cpu'}
    )
    
    # Connect to the EXACT same database built for the TOS
    vector_db = Chroma(
        persist_directory="./chroma_db_store", 
        embedding_function=embedding_model
    )
    
    # Deposit the new textbook vectors directly alongside the TOS vectors
    vector_db.add_documents(final_chunks)
    
    print("\n--- Hybrid Ingestion Complete ---")
    print("Your non-parametric memory is now fully populated and structurally sound.")


if __name__ == "__main__":
    # Define the directory where your reference PDFs are stored
    target_directory = "./textbooks"
    
    # Create the directory if it doesn't exist yet
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        print(f"Created directory '{target_directory}'. Please place your PDFs inside and run again.")
    else:
        batch_ingest_hybrid(target_directory)