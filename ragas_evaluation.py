import os
import sys
import time
import types
import json
import pandas as pd
import numpy as np

# --- FIX FOR RAGAS IMPORT BUG ---
dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat
# --------------------------------

from datasets import Dataset
from dotenv import load_dotenv

# RAGAS and Langchain integrations
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import Faithfulness

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
os.environ["TORCH_COMPILE_DISABLE"] = "1"

def load_exam_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_ragas_dataset(exam_data):
    questions = []
    answers = []
    contexts = []

    for item in exam_data:
        q_text = item.get("question", "")
        
        correct_letter = item.get("correct_answer", "")
        correct_text = item.get("options", {}).get(correct_letter, "")
        rationale = item.get("rationale", "")
        full_answer = f"{correct_text}. {rationale}"
        
        item_contexts = item.get("retrieved_contexts", [])
        if not item_contexts:
            item_contexts = ["No context provided."]
            
        questions.append(q_text)
        answers.append(full_answer)
        contexts.append(item_contexts)

    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts
    }
    return Dataset.from_dict(data)

def compute_embedding_relevancy(questions, answers, embedding_model):
    """Calculates cosine similarity between questions and answers as relevancy score."""
    q_embeds = np.array(embedding_model.embed_documents(questions))
    a_embeds = np.array(embedding_model.embed_documents(answers))
    
    # Cosine similarity calculation
    dot_products = np.sum(q_embeds * a_embeds, axis=1)
    q_norms = np.linalg.norm(q_embeds, axis=1)
    a_norms = np.linalg.norm(a_embeds, axis=1)
    
    similarities = dot_products / (q_norms * a_norms + 1e-9)
    # Scale from [-1, 1] to [0, 1]
    relevancy_scores = (similarities + 1) / 2
    return np.round(relevancy_scores, 4).tolist()

def run_evaluation(dataset, raw_exam_data):
    print("Initializing Gemini 3.5 Flash-Lite as the RAGAS Judge...")
    
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite", 
        max_retries=3
    )
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5", 
        model_kwargs={'trust_remote_code': True, 'device': 'cpu'} 
    )

    strict_rate_limit = RunConfig(max_workers=1)

    print(f"\nStarting Evaluation for {len(dataset)} items...")
    faithfulness_scores = []
    
    for i in range(len(dataset)):
        print(f"--> Evaluating Item {i+1} of {len(dataset)}...")
        single_item_dataset = dataset.select([i])
        
        try:
            result = evaluate(
                dataset=single_item_dataset,
                metrics=[Faithfulness()],
                llm=gemini_llm,
                embeddings=embedding_model,
                run_config=strict_rate_limit
            )
            score = result.to_pandas()["faithfulness"].iloc[0]
            faithfulness_scores.append(score)
        except Exception as e:
            print(f"  [!] Error on item {i+1}: {e}")
            faithfulness_scores.append(None)
            
        if i < len(dataset) - 1:
            time.sleep(20)
            
    print("\nComputing Answer Relevancy via Semantic Embedding Similarity...")
    questions = dataset["user_input"]
    answers = dataset["response"]
    relevancy_scores = compute_embedding_relevancy(questions, answers, embedding_model)
    
    # Build final DataFrame
    df = pd.DataFrame({
        "user_input": questions,
        "retrieved_contexts": dataset["retrieved_contexts"],
        "response": answers,
        "faithfulness": faithfulness_scores,
        "answer_relevancy": relevancy_scores
    })
    
    return df

if __name__ == "__main__":
    print("Step 1: Loading generated mock exam...")
    exam_json = "abnormal_psych_mock_exam.json"
    
    if not os.path.exists(exam_json):
        print(f"Error: {exam_json} not found.")
    else:
        exam_data = load_exam_data(exam_json)
        ragas_dataset = prepare_ragas_dataset(exam_data)
        
        final_dataframe = run_evaluation(ragas_dataset, exam_data)
        
        print("\n--- Evaluation Complete ---")
        output_csv = "ragas_evaluation_results.csv"
        final_dataframe.to_csv(output_csv, index=False)
        print(f"Results exported to '{output_csv}'")