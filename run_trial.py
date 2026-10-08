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
from ragas.metrics import Faithfulness
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

PRIMARY_KEY = API_KEYS[0]

# --- RAGAS Evaluator Setup ---
EVALUATOR_LLM = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.0,
    google_api_key=PRIMARY_KEY,
    candidate_count=1,
    max_retries=2,
    timeout=60
)

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'trust_remote_code': True, 'device': DEVICE},
    encode_kwargs={'normalize_embeddings': True}
)

RUN_CONFIG = RunConfig(max_workers=4)

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
    """Recursively removes the unsupported 'additionalProperties' key from a Pydantic schema."""
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
        client = genai.Client(api_key=PRIMARY_KEY)
        res = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=f"Generate a concise, alternate search query with synonyms and theorists for vector search. Subject: {subject}. Competency: {competency}. Previous: {failed_query}. Output ONLY the query string.",
            config=genai_types.GenerateContentConfig(temperature=0.3, max_output_tokens=50)
        )
        return res.text.strip().replace('"', '')
    except Exception:
        return f"{subject} {competency} core concepts principles"

def run_single_batch(subject: str, competency: str, target_blooms: str, items_to_generate: int, selected_references: list):
    base_query = build_domain_query(subject, competency)
    active_query = base_query
    attempt = 0
    max_retries = 2
    current_final_k = 7
    domain_settings = get_domain_settings(subject)

    # Extract and clean schema outside the loop
    raw_schema = AssessmentBatchSchema.model_json_schema()
    clean_schema = strip_additional_properties(raw_schema)

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
1. NO DEFINITION LEAKING: Never state the verbatim textbook definition or dead-giveaways in the stem.
2. DISTRACTOR HOMOGENEITY: All 4 options (A, B, C, D) must belong to the exact same conceptual taxonomy and have uniform length and grammatical structure.
3. THE COVER TEST: An examinee cannot guess the answer by looking at the options alone without analyzing the stem.
4. CHAIN-OF-THOUGHT MANDATE: Fill step_1_core_concept, step_2_vignette_design, and distractor_rationales for every item.
"""

        if is_hots:
            sampled = random.sample(domain_settings, min(items_to_generate, len(domain_settings)))
            settings_str = "\n".join([f"- Item {i+1}: Situate within a '{s}' context." for i, s in enumerate(sampled)])
            prompt_rules = f"\nCOGNITIVE LEVEL INSTRUCTION (HOTS - {target_blooms}):\n- Generate realistic applied scenarios.\n{settings_str}\n"
        else:
            prompt_rules = f"\nCOGNITIVE LEVEL INSTRUCTION (LOTS - {target_blooms}):\n- DIRECT CONCEPTUAL INQUIRY: Do NOT invent fictional characters or company names. Test taxonomic distinctions directly.\n"

        final_prompt = prompt_header + prompt_rules

        print(f"[Generation] Calling gemini-3.5-flash-lite for {items_to_generate} items...")
        client = genai.Client(api_key=PRIMARY_KEY)
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=final_prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=clean_schema, # Using the cleaned dictionary
                temperature=0.35
            )
        )

        parsed_batch = json.loads(response.text).get("items", [])
        if not parsed_batch:
            attempt += 1
            continue

        print(f"[Evaluation] Running RAGAS Faithfulness evaluation with Vignette Bypass...")
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

        eval_result = evaluate(
            dataset=eval_dataset,
            metrics=[Faithfulness()],
            llm=EVALUATOR_LLM,
            embeddings=EMBEDDINGS,
            run_config=RUN_CONFIG,
            raise_exceptions=True
        )

        df_scores = eval_result.to_pandas()
        batch_passed = True
        output_items = []

        for idx, item in enumerate(parsed_batch):
            f_score = float(df_scores.iloc[idx].get("faithfulness", 0.0) or 0.0)
            if math.isnan(f_score):
                f_score = 0.0

            status = "PASSED" if f_score >= 0.70 else "FLAGGED_FOR_MANUAL_REVIEW"
            if status == "FLAGGED_FOR_MANUAL_REVIEW":
                batch_passed = False

            # Format strictly matching the target JSON structure (stripping CoT keys)
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
                "evaluation_metrics": {
                    "faithfulness": round(f_score, 3)
                },
                "status": status,
                "calibration_attempts": attempt
            }
            output_items.append(clean_item)

        if batch_passed or attempt == max_retries:
            return output_items
        else:
            print(f"[Retry] Batch below threshold. Rewriting search query...")
            active_query = rewrite_search_query(subject, competency, active_query)
            current_final_k = 10
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