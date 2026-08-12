import os
import json
from google import genai
from google.genai import types
from retrieve_and_rerank import retrieve_and_rerank

def generate_mock_question(topic_query, target_blooms_level):
    print(f"Step 1: Retrieving context for '{topic_query}'...")
    # Retrieve the top 3 most relevant chunks using your existing script
    best_chunks = retrieve_and_rerank(topic_query, top_k=10, final_k=3)
    
    # Combine the chunks into a single string to act as our factual wall of evidence
    retrieved_context = "\n\n".join([chunk.page_content for chunk in best_chunks])
    
    print("\nStep 2: Connecting to Gemini API...")
    # The client automatically picks up the GEMINI_API_KEY environment variable
    client = genai.Client()
    
    # The Pentagram Prompting Framework
    prompt = f"""
    [PERSONA]
    You are an expert instructional designer and psychometrician creating professional board licensure examinations.
    
    [CONTEXT]
    You must synthesize your multiple-choice questions EXCLUSIVELY from the following retrieved syllabus and text. Do not use outside knowledge.
    <context>
    {retrieved_context}
    </context>
    
    [TASK]
    Create a highly plausible, non-repeatable multiple-choice question about: {topic_query}. 
    The question must strictly target the {target_blooms_level} level of Bloom's Taxonomy. Predict logical student errors and formulate plausible distractors based on common misconceptions.
    
    [OUTPUT]
    Return a structured JSON object containing the "question_stem", a list of four "options", the "correct_answer", and a brief "rationale".
    
    [CONSTRAINT]
    - Do not hallucinate outside the provided context.
    - Ensure the question matches the requested cognitive depth (LOTS vs HOTS).
    """
    
    print(f"Step 3: Generating question targeting {target_blooms_level}...")
    
    # We enforce JSON output and use a low temperature for strict factual adherence
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2 
        )
    )   
    
    print("\n--- Generated Assessment Item ---")
    # Parsing and printing the structured JSON response
    generated_item = json.loads(response.text)
    print(json.dumps(generated_item, indent=4))

if __name__ == "__main__":
    # A test query mapped directly to your retrieved TOS chunk
    query = "Explain the cognitive theories in the development of psychological disorders."
    generate_mock_question(query, target_blooms_level="Understanding")