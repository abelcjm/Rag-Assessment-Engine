import os
import sys
import types
import json
import warnings
from dotenv import load_dotenv

# Mock uninstalled langchain_community VertexAI module to bypass Ragas legacy import validation
dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat

warnings.filterwarnings("ignore")
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from google import genai
from google.genai import types as genai_types
from retrieve_and_rerank import retrieve_and_rerank
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
try:
    from ragas.metrics import ContextRelevance
except ImportError:
    # Support backward compatibility across older Ragas library versions
    from ragas.metrics import ContextRelevancy as ContextRelevance

from ragas.run_config import RunConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

# Evaluator LLM used exclusively for scoring validation criteria inside Ragas
EVALUATOR_LLM = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", 
    temperature=0.0,
    google_api_key=API_KEY,
    candidate_count=1,
    max_retries=3,
    timeout=60
)

# Aligned with ChromaDB embedding space to ensure semantic consistency during Ragas evaluation
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    model_kwargs={'trust_remote_code': True, 'device': 'cpu'}
)

RAGAS_METRICS = [Faithfulness(), ContextRelevance(), AnswerRelevancy(strictness=1)]
RUN_CONFIG = RunConfig(max_workers=1)

def resolve_context_metric_value(df_row):
    """
    Extracts the context relevance metric value across varying Ragas schema conventions.

    Args:
        df_row (dict): Row dictionary from the evaluated Ragas DataFrame.

    Returns:
        float: Context relevance score or 0.0 if not found.
    """
    for col in ["nv_context_relevance", "context_relevance", "context_relevancy"]:
        if col in df_row:
            return df_row[col]
    return 0.0

def generate_assessment_item(subject, competency, target_blooms_level, selected_references=["All"]):
    """
    Generates a psychometric examination item using retrieval-augmented generation and Ragas evaluation.

    If an item fails validation thresholds, the function adjusts retrieval window size (k)
    and retries up to MAX_RETRIES. Returns the highest-scoring draft if thresholds are not met.

    Args:
        subject (str): Assessment subject area (e.g., 'Abnormal Psychology').
        competency (str): Curricular competency statement to assess.
        target_blooms_level (str): Cognitive depth target (e.g., 'Understanding', 'Applying').
        selected_references (list[str]): Optional list of document sources to restrict search scope.

    Returns:
        dict: Generated item containing stem, options, rationale, citations, and evaluation metrics.
    """
    if not API_KEY:
        raise ValueError("CRITICAL: GEMINI_API_KEY not found! Check your .env file.")
    
    client = genai.Client(api_key=API_KEY)
    query = f"{subject} {competency} clinical diagnostic criteria symptoms classification definitions etiology"
    
    MAX_RETRIES = 2
    attempt = 0
    current_final_k = 5
    
    best_item = None
    best_composite_score = -1.0
    
    while attempt <= MAX_RETRIES:
        print(f"\n--- Generation Cycle (Attempt {attempt + 1}/{MAX_RETRIES + 1}) ---")
        print(f"Retrieving enriched context for: '{competency}' (final_k={current_final_k})...")
        
        best_chunks = retrieve_and_rerank(
            query=query, 
            top_k=12, 
            final_k=current_final_k, 
            reference_filter=selected_references
        )
        
        formatted_chunks, citations, raw_contexts = [], [], []
        
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
        
        print("Synthesizing item via Gemini 3.5 Flash-Lite...")
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3
            )
        )   
        
        generated_item = json.loads(response.text)
        
        print("Evaluating Draft with RAGAS (Gemini 3.1 Flash-Lite)...")
        # Structure payload to support both standard and legacy Ragas dataset interfaces
        eval_dataset = Dataset.from_dict({
            "user_input": [generated_item.get("question", "")],
            "question": [generated_item.get("question", "")],
            "response": [f"{generated_item.get('options', {}).get(generated_item.get('correct_answer', ''), '')}. {generated_item.get('rationale', '')}"],
            "answer": [f"{generated_item.get('options', {}).get(generated_item.get('correct_answer', ''), '')}. {generated_item.get('rationale', '')}"],
            "retrieved_contexts": [raw_contexts or ["No context provided."]],
            "contexts": [raw_contexts or ["No context provided."]]
        })
        
        try:
            eval_result = evaluate(
                dataset=eval_dataset,
                metrics=RAGAS_METRICS,
                llm=EVALUATOR_LLM,
                embeddings=EMBEDDINGS,
                run_config=RUN_CONFIG,
                raise_exceptions=True
            )
            scores = eval_result.to_pandas().iloc[0].to_dict()
            
            f_score = float(scores.get("faithfulness", 0.0) or 0.0)
            cr_score = float(resolve_context_metric_value(scores) or 0.0)
            ar_score = float(scores.get("answer_relevancy", 0.0) or 0.0)
            
            print(f"    -> Scores: Faithfulness: {f_score:.2f} | Context Rel: {cr_score:.2f} | Answer Rel: {ar_score:.2f}")
            
            # Record metrics and preserve the best performing attempt across retries
            composite_score = f_score + cr_score + ar_score
            generated_item["citations"] = citations
            generated_item["retrieved_contexts"] = raw_contexts
            generated_item["evaluation_metrics"] = {
                "faithfulness": round(f_score, 3),
                "context_relevance": round(cr_score, 3),
                "answer_relevancy": round(ar_score, 3)
            }
            
            if composite_score > best_composite_score:
                best_composite_score = composite_score
                best_item = generated_item.copy()
            
            # Acceptance thresholds: Faithfulness >= 0.80, Context Relevancy >= 0.70, Answer Relevancy >= 0.70
            if f_score >= 0.80 and cr_score >= 0.70 and ar_score >= 0.70:
                print("    [✓] All thresholds met. Accepting Draft.")
                generated_item["status"] = "PASSED"
                generated_item["calibration_attempts"] = attempt
                return generated_item
            else:
                print("    [X] Threshold(s) missed. Triggering Recalibration...")
                # Dynamically alter retrieval depth based on failure mode
                if cr_score < 0.70:
                    current_final_k = 3
                elif f_score < 0.80:
                    current_final_k = 7
                    
        except Exception as e:
            print(f"    [!] Error during evaluation: {e}")
            
        attempt += 1

    print(f"[!] Max retries reached. Returning highest-rated attempt (Score: {round(best_composite_score, 2)}). Flagging for manual review.")
    
    if best_item:
        best_item["status"] = "FLAGGED_FOR_MANUAL_REVIEW"
        best_item["calibration_attempts"] = attempt
        return best_item
    else:
        return {"error": "Failed to generate any valid items after max retries."}

if __name__ == "__main__":
    test_item = generate_assessment_item(
        subject="Abnormal Psychology",
        competency="Explain the cognitive theories in the development of psychological disorders.",
        target_blooms_level="Understanding"
    )
    print("\n--- Final Output ---")
    print(json.dumps(test_item, indent=4))