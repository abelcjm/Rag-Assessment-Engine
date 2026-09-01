import os
import json
from google import genai
from google.genai import types
from retrieve_and_rerank import retrieve_and_rerank
from dotenv import load_dotenv

load_dotenv()

def generate_assessment_item(subject, competency, target_blooms_level, selected_references=["All"]):
    # 1. Enriched clinical retrieval query
    query = f"{subject} {competency} clinical diagnostic criteria symptoms classification definitions etiology"
    print(f"Retrieving enriched context for: '{competency}'...")
    
    best_chunks = retrieve_and_rerank(
        query=query, 
        top_k=12, 
        final_k=5, 
        reference_filter=selected_references
    )
    
    formatted_chunks = []
    citations = []
    raw_contexts = []
    
    for chunk in best_chunks:
        source_name = chunk.metadata.get("source", "Unknown Document")
        page_num = chunk.metadata.get("page", "Unknown Page")
        content = chunk.page_content.strip()
        
        citations.append({"source": source_name, "page": page_num})
        raw_contexts.append(content)
        
        chunk_text = (
            f"--- START OF SOURCE ---\n"
            f"Document: {source_name}\n"
            f"Page: {page_num}\n"
            f"Content:\n{content}\n"
            f"--- END OF SOURCE ---"
        )
        formatted_chunks.append(chunk_text)
        
    retrieved_context = "\n\n".join(formatted_chunks) if formatted_chunks else "No context available."

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("CRITICAL: GEMINI_API_KEY not found! Check your .env file.")
    
    client = genai.Client(api_key=api_key)
    
    # Prompt structured specifically to prevent Flash-Lite hallucination
    prompt = f"""
    You are an expert psychometric test developer crafting items for the Philippine Psychometrician Licensure Examination.

    TARGET DOMAIN: {subject}
    TARGET COMPETENCY: {competency}
    TARGET COGNITIVE TIER (BLOOM'S): {target_blooms_level}

    READ THE FOLLOWING TEXTBOOK PASSAGES CAREFULLY:
    <context>
    {retrieved_context}
    </context>

    MANDATORY INSTRUCTIONS:
    1. EXTRACT EVIDENCE FIRST: Locate a specific sentence in the context that directly addresses the competency.
    2. STRICT GROUNDING: Write the question stem and the correct answer based ONLY on that extracted evidence. Do NOT add external diagnostic criteria, numbers, or theories not present in the text.
    3. RATIONALE STRUCTURE: The rationale must state the exact evidence from the text, citing the Document and Page number.
    4. OPTIONS: Provide 4 plausible, mutually exclusive options (A, B, C, D) of equal length.

    OUTPUT FORMAT:
    Return ONLY a valid JSON object matching this schema:
    {{
      "subject": "{subject}",
      "competency": "{competency}",
      "blooms_level": "{target_blooms_level}",
      "question": "Question stem here...",
      "options": {{
        "A": "Option text...",
        "B": "Option text...",
        "C": "Option text...",
        "D": "Option text..."
      }},
      "correct_answer": "A",
      "rationale": "According to [Document, Page X], '[exact quote or direct paraphrase]' which demonstrates that [explanation]."
    }}
    """
    
    print(f"Generating item targeting {target_blooms_level} via Gemini 3.5 Flash-Lite...")
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )   
    
    generated_item = json.loads(response.text)
    generated_item["citations"] = citations
    generated_item["retrieved_contexts"] = raw_contexts
    
    return generated_item

if __name__ == "__main__":
    test_item = generate_assessment_item(
        subject="Abnormal Psychology",
        competency="Explain the cognitive theories in the development of psychological disorders.",
        target_blooms_level="Understanding"
    )
    print("\n--- Generated Assessment Item ---")
    print(json.dumps(test_item, indent=4))