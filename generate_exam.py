import os
import sys
import types
import json
import warnings
import time
import math
import threading
from pydantic import BaseModel
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
    from ragas.metrics import ContextRelevancy as ContextRelevance

from ragas.run_config import RunConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# --- PYDANTIC SCHEMAS FOR STRUCTURED OUTPUT ---
class ItemOptions(BaseModel):
    A: str
    B: str
    C: str
    D: str

class AssessmentItemSchema(BaseModel):
    subject: str
    competency: str
    blooms_level: str
    question: str
    options: ItemOptions
    correct_answer: str
    rationale: str

# --- THREAD-SAFE CONCURRENCY & API KEY ROTATION MANAGER ---
raw_keys = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

if not API_KEYS:
    raise ValueError("CRITICAL: No Gemini API keys found! Set GEMINI_API_KEYS='key1,key2' in your .env file.")

CURRENT_KEY_INDEX = 0
key_lock = threading.Lock()
chroma_lock = threading.Lock()  # Protects SQLite from concurrent read collisions

def get_current_key():
    with key_lock:
        return API_KEYS[CURRENT_KEY_INDEX]

def rotate_key(failed_key=None):
    global CURRENT_KEY_INDEX, EVALUATOR_LLM
    with key_lock:
        current_active = API_KEYS[CURRENT_KEY_INDEX]
        
        # If another thread already rotated away from the failed key, reuse the new key
        if failed_key is not None and failed_key != current_active:
            print(f"\n[i] Key was already rotated by another thread to Key {CURRENT_KEY_INDEX + 1}/{len(API_KEYS)}.")
            return current_active

        CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(API_KEYS)
        new_key = API_KEYS[CURRENT_KEY_INDEX]
        print(f"\n[!] 429 Quota Exceeded. Rotating to API Key {CURRENT_KEY_INDEX + 1}/{len(API_KEYS)}...")
        
        EVALUATOR_LLM = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite", 
            temperature=0.0,
            google_api_key=new_key,
            candidate_count=1,
            max_retries=2,
            timeout=60
        )
        return new_key

# Initial Evaluator Setup
EVALUATOR_LLM = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", 
    temperature=0.0,
    google_api_key=get_current_key(),
    candidate_count=1,
    max_retries=2,
    timeout=60
)

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    model_kwargs={'trust_remote_code': True, 'device': 'cpu'}
)

RUN_CONFIG = RunConfig(max_workers=1)

def resolve_context_metric_value(df_row):
    for col in ["nv_context_relevance", "context_relevance", "context_relevancy"]:
        if col in df_row:
            return df_row[col]
    return 0.0

def generate_assessment_item(subject, competency, target_blooms_level, selected_references=["All"]):
    query = f"{subject} {competency} clinical diagnostic criteria symptoms classification definitions etiology"
    
    MAX_RETRIES = 2
    attempt = 0
    current_final_k = 5
    
    best_item = None
    best_faithfulness = -1.0
    quota_retry_count = 0
    max_quota_retries = max(len(API_KEYS) * 2, 4)
    
    while attempt <= MAX_RETRIES:
        print(f"\n--- Generation Cycle (Attempt {attempt + 1}/{MAX_RETRIES + 1}) ---")
        print(f"Retrieving context for: '{competency[:45]}...' (final_k={current_final_k})")
        
        # Thread-locked retrieval to keep SQLite safe during concurrent executions
        with chroma_lock:
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
        """
        
        try:
            active_key = get_current_key()
            client = genai.Client(api_key=active_key)
            
            print("Synthesizing item via Gemini 3.5 Flash-Lite...")
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AssessmentItemSchema,
                    temperature=0.3
                )
            )   
            
            generated_item = json.loads(response.text)
            generated_item["citations"] = citations
            generated_item["retrieved_contexts"] = raw_contexts
            
            eval_dataset = Dataset.from_dict({
                "user_input": [generated_item.get("question", "")],
                "question": [generated_item.get("question", "")],
                "response": [f"{generated_item.get('options', {}).get(generated_item.get('correct_answer', ''), '')}. {generated_item.get('rationale', '')}"],
                "answer": [f"{generated_item.get('options', {}).get(generated_item.get('correct_answer', ''), '')}. {generated_item.get('rationale', '')}"],
                "retrieved_contexts": [raw_contexts or ["No context provided."]],
                "contexts": [raw_contexts or ["No context provided."]]
            })
            
            with key_lock:
                current_evaluator = EVALUATOR_LLM

            # --- STAGE 1: FAITHFULNESS GATEKEEPER ---
            print("Evaluating Stage 1: Faithfulness...")
            eval_f = evaluate(
                dataset=eval_dataset,
                metrics=[Faithfulness()],
                llm=current_evaluator,
                embeddings=EMBEDDINGS,
                run_config=RUN_CONFIG,
                raise_exceptions=True
            )
            scores_f = eval_f.to_pandas().iloc[0].to_dict()
            f_score = float(scores_f.get("faithfulness", 0.0) or 0.0)
            
            if math.isnan(f_score):
                raise ValueError("429 Quota Exceeded during Faithfulness evaluation.")
                
            print(f"    -> Faithfulness: {f_score:.2f} (Floor: 0.75)")
            
            # Track candidate with highest Faithfulness
            if f_score > best_faithfulness:
                best_faithfulness = f_score
                best_item = generated_item.copy()
                best_item["evaluation_metrics"] = {"faithfulness": round(f_score, 3)}
                best_item["eval_dataset"] = eval_dataset

            # Check Stage 1 Floor (0.75)
            if f_score >= 0.75:
                print("    [✓] Faithfulness passed. Running Secondary Metrics...")
                eval_s = evaluate(
                    dataset=eval_dataset,
                    metrics=[ContextRelevance(), AnswerRelevancy(strictness=1)],
                    llm=current_evaluator,
                    embeddings=EMBEDDINGS,
                    run_config=RUN_CONFIG,
                    raise_exceptions=True
                )
                scores_s = eval_s.to_pandas().iloc[0].to_dict()
                cr_score = float(resolve_context_metric_value(scores_s) or 0.0)
                ar_score = float(scores_s.get("answer_relevancy", 0.0) or 0.0)
                
                if math.isnan(cr_score) or math.isnan(ar_score):
                    raise ValueError("429 Quota Exceeded during Secondary evaluation.")
                    
                print(f"    -> Context Rel: {cr_score:.2f} | Answer Rel: {ar_score:.2f}")
                
                generated_item["evaluation_metrics"] = {
                    "faithfulness": round(f_score, 3),
                    "context_relevance": round(cr_score, 3),
                    "answer_relevancy": round(ar_score, 3)
                }
                
                if cr_score >= 0.70 and ar_score >= 0.70:
                    print("    [✓] All criteria satisfied. Accepting Draft.")
                    generated_item["status"] = "PASSED"
                    generated_item["calibration_attempts"] = attempt
                    return generated_item
                else:
                    print("    [X] Secondary threshold(s) missed. Recalibrating...")
                    if cr_score < 0.70:
                        current_final_k = 3
                    attempt += 1
            else:
                print("    [X] Faithfulness below 0.75. Skipping secondary checks...")
                current_final_k = 7
                attempt += 1

        except Exception as e:
            error_msg = str(e).lower()
            if any(term in error_msg for term in ["429", "quota", "resourceexhausted", "exhausted"]):
                quota_retry_count += 1
                if quota_retry_count >= max_quota_retries:
                    print(f"    [!] All API keys exhausted. Aborting retry loop.")
                    break
                rotate_key(failed_key=active_key)
                time.sleep(3)
                continue  # Retry without incrementing attempt count
            else:
                print(f"    [!] Error during execution: {e}")
                attempt += 1

    print(f"[!] Target cycles completed. Best Faithfulness: {best_faithfulness:.2f}.")
    
    # Finalize the best-scoring item: run secondary metrics if they were skipped
    if best_item:
        metrics = best_item.get("evaluation_metrics", {})
        if "context_relevance" not in metrics:
            print("Running final secondary metrics check on best candidate...")
            try:
                with key_lock:
                    current_evaluator = EVALUATOR_LLM
                eval_final = evaluate(
                    dataset=best_item["eval_dataset"],
                    metrics=[ContextRelevance(), AnswerRelevancy(strictness=1)],
                    llm=current_evaluator,
                    embeddings=EMBEDDINGS,
                    run_config=RUN_CONFIG,
                    raise_exceptions=True
                )
                scores_fin = eval_final.to_pandas().iloc[0].to_dict()
                metrics["context_relevance"] = round(float(resolve_context_metric_value(scores_fin) or 0.0), 3)
                metrics["answer_relevancy"] = round(float(scores_fin.get("answer_relevancy", 0.0) or 0.0), 3)
            except Exception:
                metrics["context_relevance"] = 0.0
                metrics["answer_relevancy"] = 0.0
                
        best_item.pop("eval_dataset", None)
        best_item["status"] = "FLAGGED_FOR_MANUAL_REVIEW"
        best_item["calibration_attempts"] = attempt
        return best_item
    else:
        return {"error": f"Failed generation for: {competency}"}