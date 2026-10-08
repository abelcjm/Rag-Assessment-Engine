# generate_exam.py

import os
import sys
import types
import json
import warnings
import time
import math
import re
import random
import torch
import redis
from datetime import datetime, timezone
from typing import List, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from celery import shared_task
from supabase.client import create_client, Client

# LangChain community mock patch
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
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("CRITICAL: Supabase URL or Key is missing from environment variables.")

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
redis_client = redis.Redis.from_url(REDIS_URL)

# --- 1. CHAIN-OF-THOUGHT PYDANTIC SCHEMAS ---
class ItemOptions(BaseModel):
    A: str = Field(description="Option A")
    B: str = Field(description="Option B")
    C: str = Field(description="Option C")
    D: str = Field(description="Option D")

class AssessmentItemSchema(BaseModel):
    step_1_core_concept: str = Field(
        description="Isolate the specific mechanism, psychological model, or criteria being tested."
    )
    step_2_vignette_design: str = Field(
        description="Explain how the scenario or conceptual problem will test application without leaking textbook definitions."
    )
    question: str = Field(
        description="The exam question stem. Must strictly adhere to Bloom's guidelines and never leak the definition."
    )
    options: ItemOptions = Field(
        description="Four structurally balanced, homogeneous options (A, B, C, D)."
    )
    correct_answer: str = Field(
        description="The correct option key: must be exactly 'A', 'B', 'C', or 'D'."
    )
    rationale: str = Field(
        description="Textbook-grounded explanation verifying why the correct option is right, citing the source document and page."
    )
    distractor_rationales: Dict[str, str] = Field(
        description="Provide keys for the 3 incorrect options (e.g., 'A', 'B', 'C') explaining why a partially knowledgeable examinee would choose each."
    )

class AssessmentBatchSchema(BaseModel):
    items: List[AssessmentItemSchema] = Field(
        description="List of distinct, high-quality assessment items generated for the competency."
    )

# --- 2. TELEMETRY & API CONFIGURATION ---
class SupabaseTelemetryCallbackHandler(BaseCallbackHandler):
    """Intercepts LangChain LLM calls from RAGAS to log token usage."""
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        try:
            prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
            
            if response.llm_output and "token_usage" in response.llm_output:
                usage = response.llm_output["token_usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
            elif response.generations and response.generations[0]:
                first_gen = response.generations[0][0]
                if hasattr(first_gen, 'message') and hasattr(first_gen.message, 'usage_metadata'):
                    usage = first_gen.message.usage_metadata
                    if usage:
                        prompt_tokens = usage.get("input_tokens", usage.get("prompt_token_count", 0))
                        completion_tokens = usage.get("output_tokens", usage.get("candidates_token_count", 0))
                        total_tokens = usage.get("total_tokens", usage.get("total_token_count", 0))
            
            if total_tokens > 0:
                supabase_client.table("Token_Logs").insert({
                    "task_type": "Item Evaluation",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }).execute()
        except Exception as e:
            print(f"\n[TELEMETRY ERROR] Failed to log evaluation telemetry: {e}")

TELEMETRY_HANDLER = SupabaseTelemetryCallbackHandler()

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
            return current_active

        new_index = current_index + 1
        redis_client.set("gemini_key_index", new_index)
        new_key = API_KEYS[new_index % len(API_KEYS)]
        print(f"\n[!] Quota Exceeded. Rotating to API Key {new_index % len(API_KEYS) + 1}/{len(API_KEYS)}...")
        
        EVALUATOR_LLM = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite", temperature=0.0,
            google_api_key=new_key, candidate_count=1, max_retries=2, timeout=60,
            callbacks=[TELEMETRY_HANDLER]
        )
        return new_key

EVALUATOR_LLM = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", temperature=0.0,
    google_api_key=get_current_key(), candidate_count=1, max_retries=2, timeout=60,
    callbacks=[TELEMETRY_HANDLER]
)

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'trust_remote_code': True, 'device': DEVICE},
    encode_kwargs={'normalize_embeddings': True}
)

RUN_CONFIG = RunConfig(max_workers=4)

# --- 3. UTILITY FUNCTIONS ---
def get_domain_settings(subject: str) -> list:
    """Fetches the global vignette settings for a specific subject from the normalized Supabase table."""
    try:
        response = supabase_client.table("Domain_Settings").select("setting_name").eq("subject", subject).execute()
        if response.data:
            pool = [row["setting_name"] for row in response.data]
            if len(pool) > 0:
                return pool
    except Exception as e:
        print(f"[DB-WARNING] Failed to fetch Domain_Settings for {subject}: {e}")
    
    # Safe fallback if DB fetch fails or no rows exist for the subject
    return ["General workplace", "Academic setting", "Clinical setting", "Everyday social interaction"]

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
        terms = "diagnostic criteria symptoms DSM classification etiology features differential diagnosis"
    elif "industrial" in subject_lower or "organizational" in subject_lower:
        terms = "workplace behavior management organizational theory HRM leadership dynamics interventions"
    elif "assessment" in subject_lower:
        terms = "psychometrics reliability validity test administration norms scoring interpretation Classical Test Theory"
    elif "developmental" in subject_lower:
        terms = "stages cognitive moral psychosocial milestones lifespan development transitions"
    else:
        terms = "principles theories definitions concepts application constructs"
    return f"{subject} {competency} {terms}"

def rewrite_search_query(subject: str, competency: str, failed_query: str) -> str:
    """Invokes Gemini to semantically rewrite the query when context relevance is low."""
    try:
        client = genai.Client(api_key=get_current_key())
        rewrite_prompt = (
            f"You are a psychometrics information retrieval assistant.\n"
            f"Subject: {subject}\nCompetency: {competency}\nPrevious query: {failed_query}\n"
            f"Generate a concise, alternate dense search query containing related synonyms, alternate theorists, "
            f"and clinical or organizational sub-terms to retrieve better textbook context from a vector database. "
            f"Output ONLY the query string with no commentary or quotes."
        )
        res = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=rewrite_prompt,
            config=genai_types.GenerateContentConfig(temperature=0.3, max_output_tokens=50)
        )
        rewritten = res.text.strip().replace('"', '')
        return rewritten if rewritten else f"{subject} {competency} theoretical foundation models"
    except Exception:
        return f"{subject} {competency} core concepts principles models"

def save_to_supabase(item: dict, session_id: str):
    try:
        payload = {
            "exam_session_id": session_id,
            "subject": item.get("subject"),
            "competency": item.get("competency"),
            "blooms_level": item.get("blooms_level"),
            "question": item.get("question"),
            "options": item.get("options"),
            "correct_answer": item.get("correct_answer"),
            "rationale": item.get("rationale"),
            "distractor_rationales": item.get("distractor_rationales", {}),
            "cot_metadata": {
                "step_1_core_concept": item.get("step_1_core_concept", ""),
                "step_2_vignette_design": item.get("step_2_vignette_design", "")
            },
            "citations": item.get("citations", []),
            "evaluation_metrics": item.get("evaluation_metrics", {}),
            "status": item.get("status", "PASSED"),
            "calibration_attempts": item.get("calibration_attempts", 0)
        }
        response = supabase_client.table("Mock Exam Items").insert(payload).execute()
        if response.data:
            print(f"[DB-SUCCESS] Item saved with ID: {response.data[0]['id']}")
    except Exception as e:
        print(f"[DB-ERROR] Failed to save item to Supabase: {e}")

# --- 4. BATCH GENERATION CELERY TASK ---
@shared_task(bind=True)
def generate_assessment_batch_task(self, subject: str, competency: str, target_blooms_level: str,
                                   items_to_generate: int, selected_references: list, exam_session_id: str,
                                   custom_settings: list = None):
    """
    Batches question generation according to the Table of Specifications.
    Splits requests exceeding MAX_BATCH_SIZE (4) into consecutive sub-batches.
    Supports injecting a custom_settings array to override the global Domain_Settings table.
    """
    MAX_BATCH_SIZE = 4
    sub_batch_counts = []
    remaining = items_to_generate
    while remaining > 0:
        current_chunk = min(remaining, MAX_BATCH_SIZE)
        sub_batch_counts.append(current_chunk)
        remaining -= current_chunk

    all_processed_items = []
    base_query = build_domain_query(subject, competency)
    
    # Fetch global subject settings from Supabase (runs once per batch group)
    global_subject_pool = get_domain_settings(subject)

    for batch_idx, batch_count in enumerate(sub_batch_counts):
        attempt = 0
        MAX_RETRIES = 2
        active_query = base_query
        current_final_k = 7

        while attempt <= MAX_RETRIES:
            self.update_state(state='PROCESSING', meta={
                'step': 'Retrieval & Rerank',
                'batch': f"{batch_idx + 1}/{len(sub_batch_counts)}",
                'attempt': attempt + 1,
                'details': f'Retrieving top {current_final_k} chunks for {batch_count} items...'
            })

            retrieved_chunks = retrieve_and_rerank(
                query=active_query, top_k=20, final_k=current_final_k, reference_filter=selected_references
            )

            if not retrieved_chunks:
                active_query = rewrite_search_query(subject, competency, active_query)
                attempt += 1
                continue

            formatted_chunks, citations, raw_contexts = [], [], []
            for chunk in retrieved_chunks:
                s_name = chunk.metadata.get("source", "Unknown Document")
                p_num = chunk.metadata.get("page", "Unknown Page")
                body = chunk.page_content.strip()
                citations.append({"source": s_name, "page": p_num})
                raw_contexts.append(body)
                formatted_chunks.append(f"--- SOURCE: {s_name} (Page {p_num}) ---\n{body}\n--- END ---")

            context_block = "\n\n".join(formatted_chunks)

            # --- BLOOM'S BRANCHING PROMPT CONSTRUCTION ---
            is_hots = target_blooms_level.lower() in ["applying", "analyzing", "evaluating", "creating"]
            
            prompt_header = f"""You are a licensed Psychometrician and assessment specialist designing formal multiple-choice items for the Philippine Psychometrician Licensure Examination (BLEPP).

TARGET DOMAIN: {subject}
TARGET COMPETENCY: {competency}
TARGET COGNITIVE LEVEL: {target_blooms_level}
ITEMS TO GENERATE: {batch_count}

TEXTBOOK EXCERPTS:
<context>
{context_block}
</context>

STRICT PSYCHOMETRIC PRINCIPLES:
1. NO DEFINITION LEAKING: The scenario stem must describe a concrete behavior, structural condition, or practical problem. NEVER state the verbatim textbook definition or include dead-giveaway terminology in the stem.
2. DISTRACTOR HOMOGENEITY: All 4 options (A, B, C, D) must belong to the exact same conceptual taxonomy or level of abstraction. Keep options uniform in length, grammatical structure, and syntax.
3. THE COVER TEST: A prepared examinee must not be able to guess the answer by looking at the options alone without analyzing the stem, nor should the answer stand out through superficial cues.
4. CHAIN-OF-THOUGHT MANDATE: You must complete 'step_1_core_concept', 'step_2_vignette_design', and 'distractor_rationales' for every item.
"""

            if is_hots:
                # Prioritize custom_settings (teacher override), else use DB settings (admin control)
                if custom_settings and isinstance(custom_settings, list) and len(custom_settings) > 0:
                    subject_pool = custom_settings
                else:
                    subject_pool = global_subject_pool
                
                # Sample settings safely
                sampled_settings = random.choices(subject_pool, k=batch_count) if len(subject_pool) < batch_count else random.sample(subject_pool, batch_count)
                
                settings_instructions = "\n".join([f"- Item {i+1}: Situate within a '{setting}' context." for i, setting in enumerate(sampled_settings)])
                prompt_rules = f"""
COGNITIVE LEVEL INSTRUCTION (HOTS - {target_blooms_level}):
- Generate realistic, applied organizational, psychometric, or clinical scenarios.
- Assign each item to a distinct operational context:
{settings_instructions}
- Require examinees to diagnose, differentiate, or apply theoretical solutions rather than recall terms.
"""
            else:
                prompt_rules = f"""
COGNITIVE LEVEL INSTRUCTION (LOTS - {target_blooms_level}):
- DIRECT CONCEPTUAL INQUIRY: Do NOT invent fictional characters, company names, or elaborate narrative scenarios.
- Test functional distinctions, foundational assumptions, historical models, or taxonomic relationships directly and cleanly.
"""

            final_prompt = prompt_header + prompt_rules

            # --- GENERATION WITH GEMINI ---
            try:
                active_key = get_current_key()
                client = genai.Client(api_key=active_key)

                self.update_state(state='PROCESSING', meta={
                    'step': 'Generating Batch',
                    'batch': f"{batch_idx + 1}/{len(sub_batch_counts)}",
                    'attempt': attempt + 1,
                    'details': f'Drafting {batch_count} items via gemini-3.5-flash-lite...'
                })

                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=final_prompt,
                    config=genai_types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=AssessmentBatchSchema,
                        temperature=0.35
                    )
                )

                if response.usage_metadata:
                    try:
                        u = response.usage_metadata
                        supabase_client.table("Token_Logs").insert({
                            "task_type": "Exam Generation (Batch)",
                            "prompt_tokens": u.prompt_token_count,
                            "completion_tokens": u.candidates_token_count,
                            "total_tokens": u.total_token_count
                        }).execute()
                    except Exception as ex:
                        print(f"Token log failed: {ex}")

                parsed_batch = json.loads(response.text).get("items", [])
                if not parsed_batch:
                    attempt += 1
                    continue

                # --- RAGAS VIGNETTE BYPASS EVALUATION ---
                self.update_state(state='PROCESSING', meta={
                    'step': 'Evaluating Batch Quality',
                    'batch': f"{batch_idx + 1}/{len(sub_batch_counts)}",
                    'attempt': attempt + 1,
                    'details': 'Running RAGAS Faithfulness evaluation with Vignette Bypass...'
                })

                eval_questions = [f"Concept: {item.get('step_1_core_concept', '')}" for item in parsed_batch]
                eval_answers = [
                    f"Selected Option ({item.get('correct_answer')}): {item.get('options', {}).get(item.get('correct_answer'), '')}. "
                    f"Rationale: {item.get('rationale', '')}"
                    for item in parsed_batch
                ]
                eval_contexts = [raw_contexts] * len(parsed_batch)

                eval_dataset = Dataset.from_dict({
                    "question": eval_questions,
                    "user_input": eval_questions,
                    "answer": eval_answers,
                    "response": eval_answers,
                    "contexts": eval_contexts,
                    "retrieved_contexts": eval_contexts
                })

                eval_result = evaluate(
                    dataset=eval_dataset,
                    metrics=[Faithfulness()],
                    llm=EVALUATOR_LLM,
                    embeddings=EMBEDDINGS,
                    run_config=RUN_CONFIG,
                    raise_exceptions=True
                )

                df_scores = eval_result.to_pandas()

                # Process results per item
                batch_passed = True
                for idx, raw_item in enumerate(parsed_batch):
                    f_score = float(df_scores.iloc[idx].get("faithfulness", 0.0) or 0.0)
                    if math.isnan(f_score):
                        f_score = 0.0

                    raw_item["question"] = clean_question_stem(raw_item.get("question", ""))
                    raw_item["subject"] = subject
                    raw_item["competency"] = competency
                    raw_item["blooms_level"] = target_blooms_level
                    raw_item["citations"] = citations
                    raw_item["calibration_attempts"] = attempt
                    raw_item["evaluation_metrics"] = {"faithfulness": round(f_score, 3)}

                    if f_score >= 0.70:
                        raw_item["status"] = "PASSED"
                    else:
                        batch_passed = False
                        raw_item["status"] = "FLAGGED_FOR_MANUAL_REVIEW"

                if batch_passed or attempt == MAX_RETRIES:
                    for raw_item in parsed_batch:
                        save_to_supabase(raw_item, exam_session_id)
                        all_processed_items.append(raw_item)
                    break
                else:
                    active_query = rewrite_search_query(subject, competency, active_query)
                    current_final_k = 10
                    attempt += 1

            except Exception as e:
                err_str = str(e).lower()
                if any(x in err_str for x in ["429", "503", "quota", "resourceexhausted"]):
                    rotate_key(failed_key=active_key)
                    time.sleep(3)
                    continue
                else:
                    print(f"[ERROR in batch worker]: {e}")
                    attempt += 1

    return all_processed_items

# --- 5. WORKFLOW CALLBACKS & HEALTH ---
@shared_task(bind=True)
def finalize_exam_session_task(self, results, exam_session_id):
    """
    Collects results from all batches, flattens nested lists, and finalizes the exam session.
    """
    try:
        flattened_items = []
        for res in results:
            if isinstance(res, list):
                flattened_items.extend(res)
            elif isinstance(res, dict) and "question" in res:
                flattened_items.append(res)

        supabase_client.table("Exams").update({
            "global_status": "Pending", 
            "total_generated": len(flattened_items)
        }).eq("exam_id", exam_session_id).execute()

        print(f"\n[SUCCESS] Exam session {exam_session_id} finalized with {len(flattened_items)} items.")
        return {"status": "Complete", "session_id": exam_session_id, "items_count": len(flattened_items)}
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to finalize exam session: {e}")
        return {"status": "Error", "message": str(e)}

@shared_task(bind=True)
def fail_exam_session_task(self, request, exc, traceback, exam_session_id):
    try:
        supabase_client.table("Exams").update({"global_status": "Failed"}).eq("exam_id", exam_session_id).execute()
        print(f"\n[CRITICAL ERROR] Exam session {exam_session_id} failed: {exc}")
    except Exception:
        pass

@shared_task(bind=True)
def heartbeat_task(self):
    try:
        supabase_client.table("Worker_Health").upsert({
            "worker_name": "ai_celery_worker",
            "status": "operational",
            "last_heartbeat": datetime.now(timezone.utc).isoformat()
        }).execute()
    except Exception as e:
        print(f"\n[HEARTBEAT ERROR] Supabase ping failed: {e}")