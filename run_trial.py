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
from typing import List, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain community mock patch
dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

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

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("CRITICAL: Supabase URL or Key missing from .env!")

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

raw_keys = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]
if not API_KEYS:
    raise ValueError("CRITICAL: No Gemini API keys found in .env!")

current_key_idx = 0

def get_active_key():
    return API_KEYS[current_key_idx]

def rotate_eval_key():
    global EVALUATOR_LLM, current_key_idx
    current_key_idx = (current_key_idx + 1) % len(API_KEYS)
    new_key = API_KEYS[current_key_idx]
    print(f"\n[!] Rate Limit Encountered. Rotating to API Key {current_key_idx + 1}/{len(API_KEYS)}...")
    
    EVALUATOR_LLM = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.0,
        google_api_key=new_key,
        max_retries=2,
        timeout=60
    )
    return new_key

# --- Evaluator Setup ---
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

# --- Strict Pydantic Schemas ---
class ItemOptions(BaseModel):
    A: str = Field(description="Option A")
    B: str = Field(description="Option B")
    C: str = Field(description="Option C")
    D: str = Field(description="Option D")

class DistractorRationales(BaseModel):
    rationale_1: str = Field(description="Why incorrect option 1 is a plausible student misconception.")
    rationale_2: str = Field(description="Why incorrect option 2 is a plausible student misconception.")
    rationale_3: str = Field(description="Why incorrect option 3 is a plausible student misconception.")

class AssessmentItemSchema(BaseModel):
    step_1_core_concept: str = Field(description="Isolate the specific mechanism, psychological model, or criteria.")
    step_2_vignette_design: str = Field(description="Design how the scenario will test application without leaking definitions.")
    question: str = Field(description="Exam question stem.")
    options: ItemOptions = Field(description="Four balanced options (A, B, C, D).")
    correct_answer: str = Field(description="Exactly 'A', 'B', 'C', or 'D'.")
    rationale: str = Field(description="Textbook-grounded explanation citing document and page.")
    distractor_rationales: DistractorRationales = Field(description="Plausibility rationales for the 3 incorrect options.")

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

# --- DYNAMIC QUERY GENERATION (Replaces static query anchoring) ---
def generate_tailored_query(subject: str, competency: str, blooms_level: str) -> str:
    try:
        client = genai.Client(api_key=get_active_key())
        prompt = f"""You are a specialized psychometric search assistant.
Generate a concise, dense semantic search query to retrieve relevant textbook pages for this competency:
Subject: {subject}
Competency: {competency}
Cognitive Level: {blooms_level}

Requirements:
- Extract the core psychological construct, test name, ethical guideline, or statistical formula.
- Do NOT add generic filler like "Classical Test Theory" unless the competency specifically asks for it.
- Output ONLY 6 to 10 high-value keywords/terms. No commentary, no quotes.
"""
        res = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=genai_types.GenerateContentConfig(temperature=0.2, max_output_tokens=40)
        )
        query = res.text.strip().replace('"', '')
        if query:
            return f"{subject} {query}"
    except Exception as e:
        print(f"  [Query Gen Fallback]: {e}")
    
    clean_comp = re.sub(r'^[A-Z]\.\d+(\.[a-z])?\s*', '', competency)
    return f"{subject} {clean_comp}"

def rewrite_search_query(subject: str, competency: str, failed_query: str) -> str:
    try:
        client = genai.Client(api_key=get_active_key())
        res = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=f"Generate an alternate, distinct dense search query with synonyms and textbook terms for vector retrieval. Subject: {subject}. Competency: {competency}. Previous failed query: {failed_query}. Output ONLY the query string.",
            config=genai_types.GenerateContentConfig(temperature=0.4, max_output_tokens=50)
        )
        return res.text.strip().replace('"', '')
    except Exception:
        clean_comp = re.sub(r'^[A-Z]\.\d+(\.[a-z])?\s*', '', competency)
        return f"{subject} {clean_comp} clinical assessment practice"

def get_domain_settings(subject: str) -> list:
    try:
        res = supabase_client.table("Domain_Settings").select("setting_name").eq("subject", subject).execute()
        if res.data:
            pool = [row["setting_name"] for row in res.data]
            if pool:
                return pool
    except Exception:
        pass
    
    # Generic fallback based on domain
    if "industrial" in subject.lower():
        return ["Corporate HR department", "BPO / Call Center Operations", "Tertiary Hospital HR", "Manufacturing Plant"]
    return ["Clinical testing clinic", "School guidance office", "Corporate HR testing center", "Neuropsychological assessment unit"]

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

# --- BATCH RUNNER WITH GRACEFUL FLAGGING & CONTENT-ONLY TOC FILTER ---
def run_single_sub_batch(subject: str, competency: str, target_blooms: str, batch_count: int, selected_references: list, domain_settings: list):
    active_query = generate_tailored_query(subject, competency, target_blooms)
    print(f"  [Active Search Query]: \"{active_query}\"")

    attempt = 0
    max_retries = 2
    current_final_k = 8
    output_items = []

    raw_schema = AssessmentBatchSchema.model_json_schema()
    clean_schema = strip_additional_properties(raw_schema)
    absolute_words_pattern = re.compile(r'\b(always|never|completely|exclusively|entirely|absolutely|invariably|permanently|strictly|solely|purely|totally)\b', re.IGNORECASE)

    while attempt <= max_retries:
        print(f"  [Retrieval] Attempt {attempt + 1}: Querying vector DB (k={current_final_k})...")
        retrieved_chunks = retrieve_and_rerank(
            query=active_query, top_k=40, final_k=current_final_k + 4, reference_filter=selected_references
        )

        if not retrieved_chunks:
            print("  [Retrieval Warning] No chunks returned. Rewriting query...")
            active_query = rewrite_search_query(subject, competency, active_query)
            attempt += 1
            continue

        # CONTENT-ONLY TOC FILTER (No page < 20 check, preserves short documents/articles)
        formatted_chunks, citations, raw_contexts = [], [], []
        for chunk in retrieved_chunks:
            body = chunk.page_content.strip()
            
            # Drops only obvious TOC dot leaders (e.g., "Chapter 3 ............. 45")
            is_toc = bool(re.search(r'\.{4,}\s*\d+', body))
            if is_toc:
                continue

            s_name = chunk.metadata.get("source", "Unknown Document")
            p_num = chunk.metadata.get("page", "Unknown Page")
            citations.append({"source": s_name, "page": p_num})
            raw_contexts.append(body)
            formatted_chunks.append(f"--- SOURCE: {s_name} (Page {p_num}) ---\n{body}\n--- END ---")
            
            if len(formatted_chunks) >= current_final_k:
                break

        if not formatted_chunks:
            print("  [Context Warning] Retrieved chunks were mostly TOC. Broadening search...")
            active_query = rewrite_search_query(subject, competency, active_query)
            current_final_k = 12
            attempt += 1
            continue

        context_block = "\n\n".join(formatted_chunks)
        is_hots = target_blooms.lower() in ["applying", "analyzing", "evaluating", "creating"]

        prompt_header = f"""You are a licensed Psychometrician and assessment specialist designing formal multiple-choice items for the Philippine Psychometrician Licensure Examination (BLEPP).

TARGET DOMAIN: {subject}
TARGET COMPETENCY: {competency}
TARGET COGNITIVE LEVEL: {target_blooms}
ITEMS TO GENERATE: {batch_count}

TEXTBOOK EXCERPTS:
<context>
{context_block}
</context>

STRICT PSYCHOMETRIC PRINCIPLES:
1. STEM LENGTH & SYNTAX: Question stems must strictly be 2 to 4 lines maximum (under 60 words). Vary the opening syntax.
2. SUFFICIENT CONTEXT ONLY: Provide only the essential behavioral or psychometric problem. NEVER define the mechanism in the stem or give away the answer.
3. DISTRACTOR HOMOGENEITY: All 4 options (A, B, C, D) must belong to the exact same conceptual taxonomy and have uniform grammatical structure and length.
4. NO ABSOLUTE MODIFIERS (MANDATORY): Never use words such as 'completely', 'always', 'never', 'exclusively', 'invariably', 'solely', 'strictly', or 'entirely' in ANY option.
   - REJECTED (Strawman): "Adopt the test because validity completely supersedes concerns regarding test fairness."
   - ACCEPTED (Plausible Misconception): "Adopt the test because criterion validity satisfies the initial screening standards."
   - REJECTED (Strawman): "It remains acceptable if raw scores are interpreted solely by the advisor."
   - ACCEPTED (Plausible Misconception): "It remains acceptable if raw scores are corroborated by the advisor's notes."
5. CHAIN-OF-THOUGHT MANDATE: Fill step_1_core_concept, step_2_vignette_design, and distractor_rationales with plausible student misconceptions for every item.
"""

        if is_hots:
            sampled = random.sample(domain_settings, min(batch_count, len(domain_settings))) if len(domain_settings) >= batch_count else random.choices(domain_settings, k=batch_count)
            settings_str = "\n".join([f"- Item {i+1}: Situate within a '{s}' context." for i, s in enumerate(sampled)])
            prompt_rules = f"\nCOGNITIVE LEVEL INSTRUCTION (HOTS - {target_blooms}):\nGenerate applied clinical, psychometric, or educational scenarios.\n{settings_str}\n"
        else:
            prompt_rules = f"\nCOGNITIVE LEVEL INSTRUCTION (LOTS - {target_blooms}):\nDIRECT CONCEPTUAL INQUIRY: Do NOT invent fictional characters or company names. Test psychometric distinctions, assumptions, and formulas directly.\n"

        final_prompt = prompt_header + prompt_rules

        try:
            print(f"  [Generation] Requesting {batch_count} items via gemini-3.5-flash-lite...")
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

            # --- FAILSAFE VALIDATION BLOCK (Graceful Flagging, No Dropping) ---
            batch_isValid = True

            for item in parsed_batch:
                stem_words = item.get("question", "").split()
                correct_key = item.get("correct_answer", "")
                options = item.get("options", {})
                
                # Check 1: Stem Length
                if len(stem_words) > 65:
                    if attempt < max_retries:
                        batch_isValid = False
                        break
                    else:
                        item["status"] = "FLAGGED_FOR_MANUAL_REVIEW"
                        item.setdefault("evaluation_metrics", {})["regex_warning"] = "Stem exceeds word limit (65+ words)"

                # Check 2: Absolute Words Regex
                for key, text in options.items():
                    if key != correct_key and isinstance(text, str):
                        if absolute_words_pattern.search(text):
                            if attempt < max_retries:
                                batch_isValid = False
                                break
                            else:
                                item["status"] = "FLAGGED_FOR_MANUAL_REVIEW"
                                item.setdefault("evaluation_metrics", {})["regex_warning"] = f"Absolute word detected in distractor {key}"

                if not batch_isValid:
                    break

            if not batch_isValid:
                print(f"  [Validation Retry] Attempt {attempt + 1} violated stem length or distractor rules. Retrying...")
                active_query = rewrite_search_query(subject, competency, active_query)
                attempt += 1
                continue

            # --- RAGAS FAITHFULNESS EVALUATION ---
            print("  [Evaluation] Evaluating Faithfulness with RAGAS...")
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
                if attempt < max_retries:
                    attempt += 1
                    continue
                df_scores = None
            else:
                df_scores = eval_result.to_pandas()

            current_sub_batch_items = []
            batch_passed = True

            for idx, item in enumerate(parsed_batch):
                f_score = 0.0
                if df_scores is not None and not df_scores.empty:
                    f_val = df_scores.iloc[idx].get("faithfulness", 0.0)
                    f_score = float(f_val) if (f_val is not None and not math.isnan(float(f_val))) else 0.0

                # Status decision: Flag on final attempt if below threshold, never drop
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
                    "distractor_rationales": item.get("distractor_rationales", {}),
                    "cot_metadata": {
                        "step_1_core_concept": item.get("step_1_core_concept", ""),
                        "step_2_vignette_design": item.get("step_2_vignette_design", "")
                    },
                    "citations": citations,
                    "retrieved_contexts": raw_contexts,
                    "evaluation_metrics": metrics,
                    "status": status,
                    "calibration_attempts": attempt + 1
                }
                current_sub_batch_items.append(clean_item)

            if batch_passed or attempt == max_retries:
                extra_eval = safe_evaluate(dataset=eval_dataset, metrics=[AnswerRelevancy(strictness=1), ContextRelevance()], raise_exceptions=False)
                if extra_eval:
                    df_extra = extra_eval.to_pandas()
                    ar_col = next((c for c in df_extra.columns if 'answer_relevan' in c), "answer_relevancy")
                    cr_col = next((c for c in df_extra.columns if 'context_relevan' in c), "context_relevance")

                    for idx, clean_item in enumerate(current_sub_batch_items):
                        ar_score = float(df_extra.iloc[idx].get(ar_col, 0.0) or 0.0)
                        cr_score = float(df_extra.iloc[idx].get(cr_col, 0.0) or 0.0)
                        clean_item["evaluation_metrics"]["answer_relevancy"] = round(0.0 if math.isnan(ar_score) else ar_score, 3)
                        clean_item["evaluation_metrics"]["context_relevance"] = round(0.0 if math.isnan(cr_score) else cr_score, 3)

                # Return ALL items (100% quota preservation)
                return current_sub_batch_items
            else:
                print("  [Faithfulness Retry] Sub-batch below 0.70 threshold. Rewriting search query...")
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
                print(f"  [ERROR] Sub-batch failed: {e}")
                attempt += 1

    return output_items

# --- MAIN EXECUTION (Processes 100% of Blueprint Quotas) ---
def main():
    # Detect available blueprint automatically
    if os.path.exists("psychological_assessment_psychometrician_blueprint.json"):
        blueprint_file = "psychological_assessment_psychometrician_blueprint.json"
        subject = "Psychological Assessment"
    elif os.path.exists("industrial_organizational psychology_blueprint.json"):
        blueprint_file = "industrial_organizational psychology_blueprint.json"
        subject = "Industrial-Organizational Psychology"
    else:
        print("Error: No valid blueprint JSON file found in directory.")
        return

    with open(blueprint_file, "r", encoding="utf-8") as f:
        blueprint_data = json.load(f)

    # Use ["All"] to allow universal retrieval across all ingested references
    selected_references = ["All"]
    domain_settings = get_domain_settings(subject)
    total_exam_items = []

    print(f"\n========================================================")
    print(f"STARTING FULL BLEPP TRIAL RUN: {subject}")
    print(f"Blueprint File: {blueprint_file}")
    print(f"Specifications to process: {len(blueprint_data)}")
    print(f"========================================================")

    for idx, row in enumerate(blueprint_data):
        comp = row.get("competency")
        bloom = row.get("bloom_level")
        total_count = row.get("items_to_generate", 1)

        print(f"\n[{idx + 1}/{len(blueprint_data)}] Spec: {comp} | Bloom: {bloom} | Items: {total_count}")

        # Standard batching: up to 4 items per LLM call
        MAX_BATCH_SIZE = 4
        sub_batches = []
        rem = total_count
        while rem > 0:
            take = min(rem, MAX_BATCH_SIZE)
            sub_batches.append(take)
            rem -= take

        for b_idx, count in enumerate(sub_batches):
            print(f" > Processing Sub-batch {b_idx + 1}/{len(sub_batches)} ({count} items)...")
            items = run_single_sub_batch(
                subject=subject,
                competency=comp,
                target_blooms=bloom,
                batch_count=count,
                selected_references=selected_references,
                domain_settings=domain_settings
            )
            total_exam_items.extend(items)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{subject.lower().replace(' ', '_')}_trial_exam_{timestamp}.json"

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(total_exam_items, f, indent=4)

    print(f"\n[COMPLETE] Successfully generated {len(total_exam_items)} items!")
    print(f"Exported directly to: {output_filename}")

if __name__ == "__main__":
    main()