import os
import sys
import time
import types
import json
import warnings
import pandas as pd
from dotenv import load_dotenv

# --- FIX FOR RAGAS ENVIRONMENT WARNINGS AND MISSING VERTEX MODULES ---
dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TORCH_COMPILE_DISABLE"] = "1"
# ---------------------------------------------------------------------

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

INPUT_JSON_PATH = "abnormal_psych_mock_exam.json"
OUTPUT_SUMMARY_CSV = "flash_lite_ablation_results.csv"
OUTPUT_ITEM_CSV = "flash_lite_question_scores.csv"
NUM_TRIALS = 3

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

TEST_CONFIGS = [
    {
        "name": "gemini-3.1-flash-lite (strictness=1)",
        "model_name": "gemini-3.1-flash-lite",
        "strictness": 1
    },
    {
        "name": "gemini-3.5-flash-lite (strictness=1)",
        "model_name": "gemini-3.5-flash-lite",
        "strictness": 1
    }
]

def prepare_ragas_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        exam_data = json.load(f)

    questions, answers, contexts = [], [], []

    for item in exam_data:
        q_text = item.get("question", "")
        correct_letter = item.get("correct_answer", "")
        correct_text = item.get("options", {}).get(correct_letter, "")
        rationale = item.get("rationale", "")

        full_answer = f"{correct_text}. {rationale}".strip()
        item_contexts = item.get("retrieved_contexts", []) or ["No context provided."]

        questions.append(q_text)
        answers.append(full_answer)
        contexts.append(item_contexts)

    return Dataset.from_dict({
        "user_input": questions,
        "question": questions,
        "response": answers,
        "answer": answers,
        "retrieved_contexts": contexts,
        "contexts": contexts
    })

def resolve_context_metric_value(df_row):
    """Detects Ragas context relevance key variant across package releases."""
    for col in ["nv_context_relevance", "context_relevance", "context_relevancy"]:
        if col in df_row:
            return df_row[col]
    return 0.0

def run_benchmark():
    if not API_KEY:
        print("[!] Error: No GEMINI_API_KEY or GOOGLE_API_KEY detected in environment.")
        return

    if not os.path.exists(INPUT_JSON_PATH):
        print(f"[!] Error: {INPUT_JSON_PATH} not found.")
        return

    print(f"Loading assessment data from: {INPUT_JSON_PATH}")
    eval_dataset = prepare_ragas_dataset(INPUT_JSON_PATH)
    total_items = len(eval_dataset)
    print(f"Loaded {total_items} assessment items successfully.\n")

    print("Initializing Hugging Face embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    strict_run_config = RunConfig(max_workers=1, timeout=60)
    all_trial_summaries = []
    all_question_records = []

    for config in TEST_CONFIGS:
        config_label = config["name"]
        model_name = config["model_name"]
        strictness_val = config["strictness"]

        print("=" * 65)
        print(f"Starting Configuration: {config_label}")
        print("=" * 65)

        evaluator_llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.0,
            google_api_key=API_KEY,
            candidate_count=1,
            max_retries=5,
            timeout=60
        )

        metrics = [
            Faithfulness(),
            ContextRelevance(),
            AnswerRelevancy(strictness=strictness_val)
        ]

        for trial in range(1, NUM_TRIALS + 1):
            print(f"\n[{config_label}] --- Trial {trial}/{NUM_TRIALS} ---")
            trial_row_results = []

            for idx in range(total_items):
                single_row_dataset = eval_dataset.select([idx])
                success = False
                attempts = 0
                max_attempts = 3

                while not success and attempts < max_attempts:
                    try:
                        print(f"Evaluating Question {idx + 1}/{total_items}...")
                        result = evaluate(
                            dataset=single_row_dataset,
                            metrics=metrics,
                            llm=evaluator_llm,
                            embeddings=embeddings,
                            run_config=strict_run_config,
                            raise_exceptions=True
                        )
                        result_dict = result.to_pandas().iloc[0].to_dict()

                        f_val = float(result_dict.get("faithfulness", 0.0) or 0.0)
                        cr_val = float(resolve_context_metric_value(result_dict) or 0.0)
                        ar_val = float(result_dict.get("answer_relevancy", 0.0) or 0.0)

                        item_record = {
                            "configuration": config_label,
                            "trial": trial,
                            "question_index": idx + 1,
                            "question": eval_dataset["user_input"][idx],
                            "faithfulness": f_val,
                            "context_relevance": cr_val,
                            "answer_relevancy": ar_val
                        }
                        
                        all_question_records.append(item_record)
                        trial_row_results.append(item_record)
                        
                        print(f"   -> Q{idx + 1} | Faithfulness: {f_val:.3f} | Context Relevance: {cr_val:.3f} | Relevancy: {ar_val:.3f}")
                        success = True
                    except Exception as e:
                        attempts += 1
                        print(f"  [!] Exception on row {idx} (Attempt {attempts}/{max_attempts}): {type(e).__name__} - {e}")
                        time.sleep(10)

                if idx < total_items - 1:
                    time.sleep(22)

            if trial_row_results:
                trial_df = pd.DataFrame(trial_row_results)
                trial_summary = {
                    "configuration": config_label,
                    "trial": trial,
                    "faithfulness": trial_df["faithfulness"].mean(),
                    "context_relevance": trial_df["context_relevance"].mean(),
                    "answer_relevancy": trial_df["answer_relevancy"].mean(),
                }
                all_trial_summaries.append(trial_summary)
                print(
                    f"\n>>> Completed Trial {trial} Mean Scores -> "
                    f"F: {trial_summary['faithfulness']:.3f} | "
                    f"R: {trial_summary['context_relevance']:.3f} | "
                    f"AR: {trial_summary['answer_relevancy']:.3f}\n"
                )

            time.sleep(15)

    if not all_trial_summaries:
        print("No evaluation records completed.")
        return

    # Export granular per-question scores
    question_df = pd.DataFrame(all_question_records)
    question_df.to_csv(OUTPUT_ITEM_CSV, index=False)
    print(f"\nItem-level scores exported to: {OUTPUT_ITEM_CSV}")

    # Export overall trial summaries
    results_df = pd.DataFrame(all_trial_summaries)
    results_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    print(f"Trial summaries exported to: {OUTPUT_SUMMARY_CSV}")

    print("\n" + "=" * 65)
    print("ABLATION STUDY SUMMARY (Mean ± Standard Deviation)")
    print("=" * 65)

    summary_df = results_df.groupby("configuration").agg(
        faithfulness_mean=("faithfulness", "mean"),
        faithfulness_std=("faithfulness", "std"),
        context_relevance_mean=("context_relevance", "mean"),
        context_relevance_std=("context_relevance", "std"),
        answer_relevancy_mean=("answer_relevancy", "mean"),
        answer_relevancy_std=("answer_relevancy", "std"),
    ).reset_index()

    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    run_benchmark()