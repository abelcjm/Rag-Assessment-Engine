import os
import sys
import types
import json
import time
import math
import re
import random
import torch
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Dict
from dotenv import load_dotenv

# LangChain community mock patch
dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat

os.environ["TORCH_COMPILE_DISABLE"] = "1"

load_dotenv()

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
from supabase.client import create_client, Client

# --- Supabase & Device Setup ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

raw_keys = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]
if not API_KEYS:
    raise ValueError("CRITICAL: No Gemini API keys found in .env!")

# Track active key index for rotation
current_key_idx = 0

def get_active_key():
    return API_KEYS[current_key_idx]

def rotate_eval_key():
    global EVALUATOR_LLM, current_key_idx
    current_key_idx = (current_key_idx + 1) % len(API_KEYS)
    new_key = API_KEYS[current_key_idx]
    print(f"\n[!] Rate Limit Exceeded. Swapping to API Key {current_key_idx + 1}/{len(API_KEYS)}...")
    
    EVALUATOR_LLM = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.0,
        google_api_key=new_key,
        max_retries=2,
        timeout=60
    )
    return new_key

# --- RAGAS Evaluator Setup ---
EVALUATOR_LLM = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.0,
    google_api_key=get_active_key(),
    max_retries=2,
    timeout=60
)

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'trust_remote_code': True, 'device': DEVICE},
    encode_kwargs={'normalize_embeddings': True}
)

RUN_CONFIG = RunConfig(max_workers=1) 

# --- Pydantic Schemas for Generation ---
class ItemOptions(BaseModel):
    A: str = Field(description="Option A")
    B: str = Field(description="Option B")
    C: str = Field(description="Option C")
    D: str = Field(description="Option D")

class AssessmentItemSchema(BaseModel):
    step_1_core_concept: str = Field(description="Isolate the specific mechanism, psychological model, or criteria.")
    step_2_vignette_design: str = Field(description="Design how the scenario will test application without leaking definitions.")
    question: str = Field(description="Exam question stem.")
    options: ItemOptions = Field(description="Four balanced options.")
    correct_answer: str = Field(description="Exactly 'A', 'B', 'C', or 'D'.")
    rationale: str = Field(description="Textbook-grounded explanation citing document and page.")
    distractor_rationales: Dict[str, str] = Field(description="Plausibility rationales for incorrect options.")

class AssessmentBatchSchema(BaseModel):
    items: List[AssessmentItemSchema] = Field(description="List of assessment items.")

# --- Helpers ---
def strip_additional_properties(schema: dict) -> dict:
    if isinstance(schema, dict):
        schema.pop("additionalProperties", None)
        for key, list_or_dict in list(schema.items()):
            strip_additional_properties(list_or_dict)
    elif isinstance(schema, list):
        for item in schema:
            strip_additional_properties(item)
    return schema

def clean_question_stem(stem: str) -> str:
    patterns = [
        r"^(According to the (provided |retrieved )?(text|passage|context|article|document|excerpt),?\s*)",
        r"^(Based on the (provided |retrieved )?(text|passage|context|article|document|excerpt),?\s*)",
        r"^(In accordance with the (provided |retrieved )?(text|passage|context),?\s*)",
        r"^(As stated in the (provided |retrieved )?(text|passage|context),?\s*)",
        r"^(From the (provided |retrieved )?(text|passage|context),?\s*)"
    ]
    cleaned = stem.strip()
    for p in patterns:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)
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

def get_domain_settings(subject: str) -> list:
    try:
        res = supabase_client.table("Domain_Settings").select("setting_name").eq("subject", subject).execute()
        if res.data:
            pool = [row["setting_name"] for row in res.data]
            if pool:
                return pool
    except Exception:
        pass
    return ["Corporate HR department", "BPO / Call Center Operations", "Tertiary Hospital", "Manufacturing Plant"]

def rewrite_search_query(subject: str, competency: str, failed_query: str) -> str:
    try:
        client = genai.Client(api_key=get_active_key())
        res = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=f"Generate a concise, alternate search query with synonyms and theorists for vector search. Subject: {subject}. Competency: {competency}. Previous: {failed_query}. Output ONLY the query string.",
            config=genai_types.GenerateContentConfig(temperature=0.3, max_output_tokens=50)
        )
        return res.text.strip().replace('"', '')
    except Exception:
        return f"{subject} {competency} core concepts principles"

def safe_evaluate(dataset, metrics, raise_exceptions=True):
    max_rotations = len(API_KEYS)
    rotations = 0
    
    while rotations <= max_rotations:
        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=EVALUATOR_LLM,
                embeddings=EMBEDDINGS,
                run_config=RUN_CONFIG,
                raise_exceptions=True 
            )
            df = result.to_pandas()
            if df.isnull().values.any():
                raise Exception("429 Quota Exceeded (NaN detected)")
            return result
        except Exception as e:
            err_str = str(e).lower()
            if any(x in err_str for x in ["429", "quota", "resource", "nan"]):
                rotate_eval_key()
                rotations += 1
                time.sleep(3)
            else:
                if raise_exceptions:
                    raise e
                else:
                    return None
    print("[!] Failed to evaluate: Exhausted all API keys.")
    return None

def run_single_batch(subject: str, competency: str, target_blooms: str, items_to_generate: int, selected_references: list):
    base_query = build_domain_query(subject, competency)
    active_query = base_query
    attempt = 0
    max_retries = 2
    current_final_k = 7
    domain_settings = get_domain_settings(subject)

    raw_schema = AssessmentBatchSchema.model_json_schema()
    clean_schema = strip_additional_properties(raw_schema)
    absolute_words_pattern = re.compile(r'\b(always|never|completely|exclusively|entirely|absolutely|invariably|permanently|strictly|solely|purely|totally)\b', re.IGNORECASE)

    while attempt <= max_retries:
        print(f"\n[Retrieval] Attempt {attempt + 1}: Querying vector DB...")
        retrieved_chunks = retrieve_and_rerank(
            query=active_query, top_k=20, final_k=current_final_k, reference_filter=selected_references
        )

        if not retrieved_chunks:
            print("[Retrieval Warning] No chunks returned. Rewriting query...")
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
        is_hots = target_blooms.lower() in ["applying", "analyzing", "evaluating", "creating"]

        prompt_header = f"""You are a licensed Psychometrician and assessment specialist designing formal multiple-choice items for the Philippine Psychometrician Licensure Examination (BLEPP).

TARGET DOMAIN: {subject}
TARGET COMPETENCY: {competency}
TARGET COGNITIVE LEVEL: {target_blooms}
ITEMS TO GENERATE: {items_to_generate}

TEXTBOOK EXCERPTS:
<context>
{context_block}
</context>

STRICT PSYCHOMETRIC PRINCIPLES:
1. STEM LENGTH & SYNTAX: Question stems must strictly be 2 to 4 lines maximum (under 60 words). Vary the opening syntax (do not start every item with "A manager at...").
2. SUFFICIENT CONTEXT ONLY: Provide only the essential behavioral or structural problem. NEVER over-explain the scenario to the point that it gives away the answer. The examinee must infer the theory, not match keywords.
3. DISTRACTOR HOMOGENEITY: All 4 options (A, B, C, D) must belong to the exact same conceptual taxonomy and have uniform length and grammatical structure.
4. NO ABSOLUTE MODIFIERS: Never use words such as 'completely', 'always', 'never', 'exclusively', 'invariably', 'solely', or 'entirely' in distractors.
5. CHAIN-OF-THOUGHT MANDATE: Fill step_1_core_concept, step_2_vignette_design, and distractor_rationales for every item.
"""

        if is_hots:
            sampled = random.sample(domain_settings, min(items_to_generate, len(domain_settings)))
            settings_str = "\n".join([f"- Item {i+1}: Situate within a '{s}' context." for i, s in enumerate(sampled)])
            prompt_rules = f"\nCOGNITIVE LEVEL INSTRUCTION (HOTS - {target_blooms}):\n- Generate realistic applied scenarios.\n{settings_str}\n"
        else:
            prompt_rules = f"\nCOGNITIVE LEVEL INSTRUCTION (LOTS - {target_blooms}):\n- DIRECT CONCEPTUAL INQUIRY: Do NOT invent fictional characters or company names. Test taxonomic distinctions directly.\n"

        final_prompt = prompt_header + prompt_rules

        try:
            print(f"[Generation] Calling gemini-3.5-flash-lite for {items_to_generate} items...")
            client = genai.Client(api_key=get_active_key())
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=final_prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=clean_schema, 
                    temperature=0.35
                )
            )

            parsed_batch = json.loads(response.text).get("items", [])
            if not parsed_batch:
                attempt += 1
                continue

            # --- FAILSAFE VALIDATION BLOCK ---
            batch_isValid = True
            for item in parsed_batch:
                stem_words = item.get("question", "").split()
                correct_key = item.get("correct_answer", "")
                options = item.get("options", {})
                
                if len(stem_words) > 65:
                    print(f"[Validation Failed] Stem too long ({len(stem_words)} words).")
                    if attempt < max_retries:
                        batch_isValid = False
                        break
                    else:
                        item["status"] = "FLAGGED_FOR_MANUAL_REVIEW"
                        item.setdefault("evaluation_metrics", {})["regex_warning"] = "Stem exceeds word limit"

                for key, text in options.items():
                    if key != correct_key and isinstance(text, str):
                        if absolute_words_pattern.search(text):
                            print(f"[Validation Failed] Absolute word found in distractor {key}.")
                            if attempt < max_retries:
                                batch_isValid = False
                                break
                            else:
                                item["status"] = "FLAGGED_FOR_MANUAL_REVIEW"
                                item.setdefault("evaluation_metrics", {})["regex_warning"] = "Absolute word detected in distractor"
                
                if not batch_isValid:
                    break

            if not batch_isValid:
                print(f"[Retry] Validation failed. Rewriting query...")
                active_query = rewrite_search_query(subject, competency, active_query)
                attempt += 1
                continue
            # --- END FAILSAFE VALIDATION BLOCK ---

            print(f"[Evaluation] Running Primary RAGAS Faithfulness check...")
            eval_questions = [f"Concept: {item.get('step_1_core_concept', '')}" for item in parsed_batch]
            eval_answers = [
                f"Selected Option ({item.get('correct_answer')}): {item.get('options', {}).get(item.get('correct_answer'), '')}. Rationale: {item.get('rationale', '')}"
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

            eval_result = safe_evaluate(dataset=eval_dataset, metrics=[Faithfulness()], raise_exceptions=True)
            if not eval_result:
                attempt += 1
                continue

            df_scores = eval_result.to_pandas()
            batch_passed = True
            output_items = []

            for idx, item in enumerate(parsed_batch):
                f_score = float(df_scores.iloc[idx].get("faithfulness", 0.0) or 0.0)
                if math.isnan(f_score):
                    f_score = 0.0

                # Only mark as PASSED if it wasn't flagged by the regex failsafe
                if f_score >= 0.70 and item.get("status") != "FLAGGED_FOR_MANUAL_REVIEW":
                    status = "PASSED"
                else:
                    status = "FLAGGED_FOR_MANUAL_REVIEW"
                    if f_score < 0.70:
                        batch_passed = False

                metrics = item.get("evaluation_metrics", {})
                metrics["faithfulness"] = round(f_score, 3)

                clean_item = {
                    "subject": subject,
                    "competency": competency,
                    "blooms_level": target_blooms,
                    "question": clean_question_stem(item.get("question", "")),
                    "options": item.get("options", {}),
                    "correct_answer": item.get("correct_answer", ""),
                    "rationale": item.get("rationale", ""),
                    "citations": citations,
                    "retrieved_contexts": raw_contexts,
                    "evaluation_metrics": metrics,
                    "status": status,
                    "calibration_attempts": attempt + 1
                }
                output_items.append(clean_item)

            if batch_passed or attempt == max_retries:
                print(f"[Evaluation] Batch completed. Running secondary metrics: Answer Relevancy & Context Relevance...")
                extra_eval = safe_evaluate(dataset=eval_dataset, metrics=[AnswerRelevancy(strictness=1), ContextRelevance()], raise_exceptions=False)
                
                if extra_eval:
                    df_extra = extra_eval.to_pandas()
                    ar_col = next((c for c in df_extra.columns if 'answer_relevan' in c), "answer_relevancy")
                    cr_col = next((c for c in df_extra.columns if 'context_relevan' in c), "context_relevance")

                    for idx, clean_item in enumerate(output_items):
                        ar_score = float(df_extra.iloc[idx].get(ar_col, 0.0) or 0.0)
                        cr_score = float(df_extra.iloc[idx].get(cr_col, 0.0) or 0.0)
                        if math.isnan(ar_score): ar_score = 0.0
                        if math.isnan(cr_score): cr_score = 0.0
                        
                        clean_item["evaluation_metrics"]["answer_relevancy"] = round(ar_score, 3)
                        clean_item["evaluation_metrics"]["context_relevance"] = round(cr_score, 3)

                return output_items
            else:
                print(f"[Retry] Batch below threshold. Rewriting search query...")
                active_query = rewrite_search_query(subject, competency, active_query)
                current_final_k = 10
                attempt += 1

        except Exception as e:
            err_str = str(e).lower()
            if any(x in err_str for x in ["429", "quota", "resourceexhausted"]):
                rotate_eval_key()
                time.sleep(3)
                continue
            else:
                print(f"[ERROR] Generation failed: {e}")
                attempt += 1

    return output_items


def main():
    blueprint_file = "industrial_organizational psychology_blueprint.json"
    if not os.path.exists(blueprint_file):
        print(f"Error: Could not find {blueprint_file} in current directory.")
        return

    with open(blueprint_file, "r", encoding="utf-8") as f:
        blueprint_data = json.load(f)

    # For a trial run, pick the first 3 requirements or run the whole file
    trial_slice = blueprint_data[:3]
    total_exam_items = []

    print(f"Starting Trial Run across {len(trial_slice)} blueprint specifications...")

    for row in trial_slice:
        comp = row.get("competency")
        bloom = row.get("bloom_level")
        count = row.get("items_to_generate", 1)
        
        print(f"\n==========================================")
        print(f"Target: {comp} ({bloom}) | Count: {count}")
        print(f"==========================================")

        items = run_single_batch(
            subject="Industrial-Organizational Psychology",
            competency=comp,
            target_blooms=bloom,
            items_to_generate=count,
            selected_references=["All"]
        )
        total_exam_items.extend(items)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"industrial_organizational_psychology_mock_exam_{timestamp}.json"

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(total_exam_items, f, indent=4)

    print(f"\n[COMPLETE] Successfully generated {len(total_exam_items)} items!")
    print(f"Exported directly to: {output_filename}")


if __name__ == "__main__":
    main()