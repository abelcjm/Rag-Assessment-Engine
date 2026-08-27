import os
import json
from google import genai
from google.genai import types
from retrieve_and_rerank import retrieve_and_rerank
from dotenv import load_dotenv

load_dotenv()

def generate_assessment_item(subject, competency, target_blooms_level, selected_references=["All"]):
    print(f"Step 1: Retrieving context for '{competency}'...")
    query = f"{subject}: {competency}"
    best_chunks = retrieve_and_rerank(query, top_k=5, final_k=3, reference_filter=selected_references)
    
    # Weave the citations directly into the text chunks
    formatted_chunks = []
    citations = []
    
    for chunk in best_chunks:
        source_name = chunk.metadata.get("source", "Unknown Document")
        page_num = chunk.metadata.get("page", "Unknown Page")
        
        citations.append({"source": source_name, "page": page_num})
        
        chunk_text = (
            f"--- START OF SOURCE ---\n"
            f"Document: {source_name}\n"
            f"Page: {page_num}\n"
            f"Content:\n{chunk.page_content}\n"
            f"--- END OF SOURCE ---"
        )
        formatted_chunks.append(chunk_text)
        
    retrieved_context = "\n\n".join(formatted_chunks)

    print("\nStep 2: Connecting to Gemini API...")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("CRITICAL: GEMINI_API_KEY not found! Check your .env file.")
    
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    [PERSONA]
    You are an expert psychometric test developer crafting items for the Philippine Psychometrician Licensure Examination.

    [CONTEXT]
    TARGET DOMAIN: {subject}
    TARGET COMPETENCY: {competency}
    TARGET COGNITIVE TIER (BLOOM'S): {target_blooms_level}

    STRICT GROUNDING RULES:
    1. SOURCE DEPENDENCE: Derive the stem, correct answer, and distractors EXCLUSIVELY from the provided textbook context below. Do not use outside knowledge.
    2. COGNITIVE ALIGNMENT: Align the question's difficulty with the provided Bloom's level.
    3. PSYCHOMETRIC STANDARDS: Every option must be mutually exclusive and homogeneous in length.
    4. RATIONALE CITATIONS: When writing the rationale, explicitly mention the Document and Page number provided in the source headers to validate your reasoning.

    <context>
    {retrieved_context}
    </context>

    [OUTPUT]
    Return a structured JSON object exactly matching this schema:
    {{
      "subject": "{subject}",
      "competency": "{competency}",
      "blooms_level": "{target_blooms_level}",
      "question": "The question stem...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "correct_answer": "A",
      "rationale": "Explanation based on the text..."
    }}
    """
    
    print(f"Step 3: Generating item targeting {target_blooms_level}...")
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2 
        )
    )   
    
    generated_item = json.loads(response.text)
    generated_item["citations"] = citations
    
    return generated_item

if __name__ == "__main__":
    test_item = generate_assessment_item(
        subject="Abnormal Psychology",
        competency="Explain the cognitive theories in the development of psychological disorders.",
        target_blooms_level="Understanding"
    )
    print("\n--- Generated Assessment Item ---")
    print(json.dumps(test_item, indent=4))