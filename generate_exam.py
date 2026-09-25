# generate_exam.py
import os
import sys
import types
import json
import warnings
import time
import math
import re
import torch
import redis
from pydantic import BaseModel
from dotenv import load_dotenv
from celery import shared_task
from supabase.client import create_client, Client

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

# --- INFRASTRUCTURE CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("CRITICAL: Supabase URL or Key is missing from environment variables.")

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
redis_client = redis.Redis.from_url(REDIS_URL)

BLUEPRINT_MAP = {
    "1": {"subject": "Abnormal Psychology", "file": "abnormal_psychology_blueprint.json"},
    "2": {"subject": "Industrial-Organizational Psychology", "file": "industrial_organizational psychology_blueprint.json"},
    "3": {"subject": "Psychological Assessment", "file": "psychological_assessment_psychometrician_blueprint.json"},
    "4": {"subject": "Developmental Psychology", "file": "developmental_psychology_blueprint.json"}
}

# --- PYDANTIC SCHEMAS ---
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

# --- REDIS-BACKED CONCURRENCY & API KEY ROTATION ---
raw_keys = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

if not API_KEYS:
    raise ValueError("CRITICAL: No Gemini API keys found! Set GEMINI_API_KEYS='key1,key2' in your .env file.")

def get_current_key():
    index = int(redis_client.get("gemini_key_index") or 0)
    return API_KEYS[index % len(API_KEYS)]

def rotate_key(failed_key=None):
    global EVALUATOR_LLM
    
    with redis_client.lock("key_rotation_lock", timeout=10):
        current_index = int(redis_client.get("gemini_key_index") or 0)
        current_active = API_KEYS[current_index % len(API_KEYS)]
        
        if failed_key is not None and failed_key != current_active:
            print(f"\n[i] Key was already rotated to {current_index % len(API_KEYS) + 1}/{len(API_KEYS)} by another worker.")
            return current_active

        new_index = current_index + 1
        redis_client.set("gemini_key_index", new_index)
        new_key = API_KEYS[new_index % len(API_KEYS)]
        print(f"\n[!] Quota Exceeded. Rotating to API Key {new_index % len(API_KEYS) + 1}/{len(API_KEYS)}...")
        
        EVALUATOR_LLM = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite", temperature=0.0,
            google_api_key=new_key, candidate_count=1, max_retries=2, timeout=60
        )
        return new_key

EVALUATOR_LLM = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", temperature=0.0,
    google_api_key=get_current_key(), candidate_count=1, max_retries=2, timeout=60
)

# --- EMBEDDING & EVALUATION SETTINGS ---
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'trust_remote_code': True, 'device': DEVICE},
    encode_kwargs={'normalize_embeddings': True}
)

RUN_CONFIG = RunConfig(max_workers=4)

def resolve_context_metric_value(df_row):
    for col in ["nv_context_relevance", "context_relevance", "context_relevancy"]:
        if col in df_row:
            return df_row[col]
    return 0.0

def clean_question_stem(stem: str) -> str:
    patterns = [
        r"^(According to the (provided |retrieved )?(text|passage|context|article|document|excerpt),?\s*)",
        r"^(Based on the (provided |retrieved )?(text|passage|context|article|document|excerpt),?\s*)",
        r"^(In accordance with the (provided |retrieved )?(text|passage|context),?\s*)",
        r"^(As stated in the (provided |retrieved )?(text|passage|context),?\s*)",
        r"^(From the (provided |retrieved )?(text|passage|context),?\s*)"
    ]
    cleaned = stem.strip()
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned.strip()

def build_domain_query(subject: str, competency: str) -> str:
    subject_lower = subject.lower()
    if "abnormal" in subject_lower:
        terms = "diagnostic criteria symptoms DSM classification etiology features"
    elif "industrial" in subject_lower or "organizational" in subject_lower:
        terms = "workplace behavior management organizational theory HRM leadership dynamics"
    elif "assessment" in subject_lower:
        terms = "psychometrics reliability validity test administration norms scoring interpretation"
    elif "developmental" in subject_lower:
        terms = "stages cognitive moral psychosocial milestones lifespan development"
    else:
        terms = "principles theories definitions concepts application"
    return f"{subject} {competency} {terms}"

def save_to_supabase(item: dict, session_id: str):
    try:
        payload = item.copy()
        payload["exam_session_id"] = session_id
        response = supabase_client.table("Mock Exam Items").insert(payload).execute()
        if response.data:
            print(f"\n[DB-SUCCESS] Item saved to Supabase with ID: {response.data[0]['id']}")
    except Exception as e:
        print(f"\n[DB-ERROR] Failed to save to Supabase: {e}")

def execute_core_generation_loop(subject, competency, target_blooms_level, selected_references, exam_session_id, celery_task=None):
    """
    Extracted core logic to allow execution by both single-item and batch-processing tasks.
    """
    query = build_domain_query(subject, competency)
    
    MAX_RETRIES = 2
    attempt = 0
    current_final_k = 5
    
    best_item = None
    best_faithfulness = -1.0
    quota_retry_count = 0
    max_quota_retries = max(len(API_KEYS) * 2, 4)
    
    while attempt <= MAX_RETRIES:
        if celery_task:
            celery_task.update_state(state='PROCESSING', meta={
                'step': 'Retrieving Context', 'attempt': attempt + 1, 'details': f'Fetching {current_final_k} chunks from vector database...'
            })
        
        best_chunks = retrieve_and_rerank(
            query=query, top_k=15, final_k=current_final_k, reference_filter=selected_references
        )
        
        formatted_chunks, citations, raw_contexts = [], [], []
        for chunk in best_chunks:
            source_name = chunk.metadata.get("source", "Unknown Document")
            page_num = chunk.metadata.get("page", "Unknown Page")
            content = chunk.page_content.strip()
            
            citations.append({"source": source_name, "page": page_num})
            raw_contexts.append(content)
            
            chunk_text = (f"--- START OF SOURCE ---\nDocument: {source_name}\nPage: {page_num}\nContent:\n{content}\n--- END OF SOURCE ---")
            formatted_chunks.append(chunk_text)
            
        retrieved_context = "\n\n".join(formatted_chunks) if formatted_chunks else "No context available."
        
        prompt = f"""You are a licensed Psychometrician and psychometric test developer creating formal multiple-choice items for the Philippine Psychometrician Licensure Examination (BLEPP).

TARGET DOMAIN: {subject}
TARGET COMPETENCY: {competency}
TARGET COGNITIVE LEVEL (BLOOM'S): {target_blooms_level}

SOURCE MATERIAL (TEXTBOOK EXCERPTS):
<context>
{retrieved_context}
</context>

ITEM WRITING RULES (STRICT COMPLIANCE REQUIRED):
1. NO SOURCE REFERENCING IN STEM:
   - NEVER start or include phrases like "According to the text", "Based on the provided passage", "As discussed in the excerpt", or "According to the author".
   - Treat the domain concepts as established, objective facts or standard scientific principles.
   - The question must stand alone as an authentic exam item. The examinee does not have access to any reference text.

2. AUTHENTIC SCENARIOS & VIGNETTES:
   - For application/analysis levels, construct a realistic workplace, clinical, or testing scenario.
   - Vary the organizations, industries, and names used (e.g., healthcare, education, retail, tech, public sector). Avoid generic clichés.

3. STRICT OPTION HOMOGENEITY:
   - All 4 options (A, B, C, D) MUST be approximately equal in length, grammatical structure, and complexity.
   - Do NOT make the correct answer significantly longer or more detailed than the distractors.
   - Distractors must be plausible, mutually exclusive alternatives within the same conceptual category.

4. GROUNDING & RATIONALE:
   - The item's factual truth must be grounded in the context provided.
   - The 'rationale' field must cite the exact Document name and Page number where the answer is verified.
"""
        
        try:
            active_key = get_current_key()
            client = genai.Client(api_key=active_key)
            
            if celery_task:
                celery_task.update_state(state='PROCESSING', meta={
                    'step': 'Generating Draft', 'attempt': attempt + 1, 'details': 'Drafting question via Gemini...'
                })
            
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AssessmentItemSchema,
                    temperature=0.4
                )
            )   
            
            generated_item = json.loads(response.text)
            
            generated_item["question"] = clean_question_stem(generated_item["question"])
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
            
            current_evaluator = EVALUATOR_LLM

            if celery_task:
                celery_task.update_state(state='PROCESSING', meta={
                    'step': 'Evaluating Faithfulness', 'attempt': attempt + 1, 'details': 'Running RAGAS hallucination checks...'
                })
            
            eval_f = evaluate(
                dataset=eval_dataset, metrics=[Faithfulness()], llm=current_evaluator,
                embeddings=EMBEDDINGS, run_config=RUN_CONFIG, raise_exceptions=True
            )
            scores_f = eval_f.to_pandas().iloc[0].to_dict()
            f_score = float(scores_f.get("faithfulness", 0.0) or 0.0)
            
            if math.isnan(f_score):
                raise ValueError("429/503 API Unavailable or Quota Exceeded during Faithfulness evaluation.")
                
            if f_score > best_faithfulness:
                best_faithfulness = f_score
                best_item = generated_item.copy()
                best_item["evaluation_metrics"] = {"faithfulness": round(f_score, 3)}
                best_item["eval_dataset"] = eval_dataset

            if f_score >= 0.75:
                if celery_task:
                    celery_task.update_state(state='PROCESSING', meta={
                        'step': 'Evaluating Relevancy', 'attempt': attempt + 1, 'details': f'Faithfulness passed ({f_score:.2f}). Checking Relevancy...'
                    })
                eval_s = evaluate(
                    dataset=eval_dataset, metrics=[ContextRelevance(), AnswerRelevancy(strictness=1)],
                    llm=current_evaluator, embeddings=EMBEDDINGS, run_config=RUN_CONFIG, raise_exceptions=True
                )
                scores_s = eval_s.to_pandas().iloc[0].to_dict()
                cr_score = float(resolve_context_metric_value(scores_s) or 0.0)
                ar_score = float(scores_s.get("answer_relevancy", 0.0) or 0.0)
                
                if math.isnan(cr_score) or math.isnan(ar_score):
                    raise ValueError("429/503 API Unavailable or Quota Exceeded during Secondary evaluation.")
                    
                generated_item["evaluation_metrics"] = {
                    "faithfulness": round(f_score, 3), "context_relevance": round(cr_score, 3), "answer_relevancy": round(ar_score, 3)
                }
                
                if cr_score >= 0.70 and ar_score >= 0.70:
                    generated_item["status"] = "PASSED"
                    generated_item["calibration_attempts"] = attempt
                    save_to_supabase(generated_item, exam_session_id)
                    return generated_item
                else:
                    if cr_score < 0.70: current_final_k = 3
                    attempt += 1
            else:
                current_final_k = 7
                attempt += 1

        except Exception as e:
            error_msg = str(e).lower()
            if any(term in error_msg for term in ["429", "503", "504", "quota", "resourceexhausted", "exhausted", "unavailable", "deadline"]):
                quota_retry_count += 1
                if quota_retry_count >= max_quota_retries: break
                rotate_key(failed_key=active_key)
                time.sleep(3)
                continue 
            else:
                attempt += 1

    if best_item:
        metrics = best_item.get("evaluation_metrics", {})
        if "context_relevance" not in metrics:
            try:
                current_evaluator = EVALUATOR_LLM
                eval_final = evaluate(
                    dataset=best_item["eval_dataset"], metrics=[ContextRelevance(), AnswerRelevancy(strictness=1)],
                    llm=current_evaluator, embeddings=EMBEDDINGS, run_config=RUN_CONFIG, raise_exceptions=True
                )
                scores_fin = eval_final.to_pandas().iloc[0].to_dict()
                
                cr_raw = float(resolve_context_metric_value(scores_fin) or 0.0)
                ar_raw = float(scores_fin.get("answer_relevancy", 0.0) or 0.0)
                
                metrics["context_relevance"] = 0.0 if math.isnan(cr_raw) else round(cr_raw, 3)
                metrics["answer_relevancy"] = 0.0 if math.isnan(ar_raw) else round(ar_raw, 3)
            except Exception:
                metrics["context_relevance"] = 0.0
                metrics["answer_relevancy"] = 0.0
                
        best_item.pop("eval_dataset", None)
        best_item["status"] = "FLAGGED_FOR_MANUAL_REVIEW"
        best_item["calibration_attempts"] = attempt
        save_to_supabase(best_item, exam_session_id)
        return best_item
    else:
        return {"error": f"Failed generation for: {competency}"}

@shared_task(bind=True)
def generate_assessment_item_task(self, subject, competency, target_blooms_level, selected_references=["All"], exam_session_id=None):
    """Fallback single-item generation for the regenerate endpoint."""
    return execute_core_generation_loop(
        subject, competency, target_blooms_level, selected_references, exam_session_id, celery_task=self
    )

@shared_task(bind=True)
def generate_blueprint_batch_task(self, exam_session_id, blueprint_id, selected_pdfs):
    """Executes a full blueprint generation loop asynchronously."""
    try:
        blueprint_config = BLUEPRINT_MAP.get(str(blueprint_id))
        if not blueprint_config or not os.path.exists(blueprint_config["file"]):
            raise ValueError(f"Invalid blueprint ID or missing file for {blueprint_id}")

        with open(blueprint_config["file"], 'r', encoding='utf-8') as f:
            blueprint_data = json.load(f)

        for requirement in blueprint_data:
            count = requirement.get("items_to_generate", 0)
            for _ in range(count):
                execute_core_generation_loop(
                    subject=blueprint_config["subject"],
                    competency=requirement["competency"],
                    target_blooms_level=requirement["bloom_level"],
                    selected_references=selected_pdfs,
                    exam_session_id=exam_session_id,
                    celery_task=self
                )
                
        supabase_client.table("Exams").update({"global_status": "Pending"}).eq("id", exam_session_id).execute()
        return {"status": "Success", "session_id": exam_session_id}

    except Exception as e:
        error_details = str(e)
        print(f"\n[CRITICAL ERROR] Blueprint batch failed: {error_details}")
        supabase_client.table("Exams").update({"global_status": "Failed"}).eq("id", exam_session_id).execute()
        return {"status": "Failed", "error": error_details}

@shared_task(bind=True)
def generate_custom_batch_task(self, exam_session_id, items, selected_pdfs):
    """Executes custom parameter generation loops asynchronously."""
    try:
        for req in items:
            count = req.get("count", 1)
            for _ in range(count):
                execute_core_generation_loop(
                    subject=req["subject"],
                    competency=req["competency"],
                    target_blooms_level=req["bloom"],
                    selected_references=selected_pdfs,
                    exam_session_id=exam_session_id,
                    celery_task=self
                )
                
        supabase_client.table("Exams").update({"global_status": "Pending"}).eq("id", exam_session_id).execute()
        return {"status": "Success", "session_id": exam_session_id}

    except Exception as e:
        error_details = str(e)
        print(f"\n[CRITICAL ERROR] Custom batch failed: {error_details}")
        supabase_client.table("Exams").update({"global_status": "Failed"}).eq("id", exam_session_id).execute()
        return {"status": "Failed", "error": error_details}