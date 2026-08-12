import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from docling.document_converter import DocumentConverter

def ingest_prc_tos(file_path):
    print(f"Initializing layout-aware ingestion for: {file_path}...")
    
    # Initialize the Docling DocumentConverter
    # This automatically handles the VLM processing to preserve spatial formatting
    converter = DocumentConverter()
    
    # Execute the conversion on the PDF
    result = converter.convert(file_path)
    
    # Export the parsed document to Markdown
    # This utilizes Docling's unified representation to preserve tables and structural dependencies
    markdown_output = result.document.export_to_markdown()
    
    print("\n--- Extraction Complete ---")
    
    # Save the output to a local Markdown file for inspection
    output_filename = "Parsed_PRC_TOS.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(markdown_output)
        
    print(f"Successfully saved semantic chunks to {output_filename}")

if __name__ == "__main__":
    # Ensure you have your PRC TOS PDF in the same directory, or update this path
    pdf_path = "TOS_Psychometricians.pdf"
    ingest_prc_tos(pdf_path)