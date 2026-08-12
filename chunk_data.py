from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def chunk_tos_markdown(file_path):
    # Loads the parsed markdown file into memory
    with open(file_path, 'r', encoding='utf-8') as f:
        markdown_document = f.read()

    # Defines the markdown headers to split on to preserve hierarchical context
    headers_to_split_on = [
        ("##", "Section_Header"),
    ]
    
    # Executes the initial split by headers
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)

    # Enforces the maximum threshold of 1000 characters and 150-character overlap
    chunk_size = 1000
    chunk_overlap = 150 
    
    # Prioritizes splitting by tables and paragraphs first to maintain structural integrity
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "|", " ", ""] 
    )

    final_chunks = text_splitter.split_documents(md_header_splits)
    
    print(f"Successfully divided the document into {len(final_chunks)} semantic chunks.")
    return final_chunks

if __name__ == "__main__":
    md_path = "Parsed_PRC_TOS.md"
    chunks = chunk_tos_markdown(md_path)
    
    print("\n--- Sample Chunk ---")
    print(chunks[0].page_content)
    print("\nMetadata:", chunks[0].metadata)