import json
import os
import time
import concurrent.futures
from datetime import datetime
from dotenv import load_dotenv

from generate_exam import generate_assessment_item

load_dotenv()

BLUEPRINT_MAP = {
    "1": {"subject": "Abnormal Psychology", "file": "abnormal_pyschology_blueprint.json"},
    "2": {"subject": "Industrial-Organizational Psychology", "file": "industrial_organizational psychology_blueprint.json"},
    "3": {"subject": "Psychological Assessment", "file": "psychological_assessment_psychometrician_blueprint.json"},
    "4": {"subject": "Developmental Psychology", "file": "developmental_psychology_blueprint.json"}
}

def generate_unique_filename(base_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.json"

def normalize_item(item):
    if hasattr(item, 'model_dump'):
        return item.model_dump()
    elif hasattr(item, 'dict'):
        return item.dict()
    return dict(item)

def process_single_item(task_id, total_tasks, subject, competency, bloom, refs):
    print(f"\n[Worker {task_id}/{total_tasks}] Dispatching: {competency[:40]}... ({bloom})")
    try:
        item = generate_assessment_item(
            subject=subject,
            competency=competency,
            target_blooms_level=bloom,
            selected_references=refs
        )
        print(f"[Worker {task_id}/{total_tasks}] Item complete.")
        return normalize_item(item)
    except Exception as e:
        print(f"[Worker {task_id}/{total_tasks}] [!] Error: {e}")
        return None

def run_batch_generation(blueprint_path, target_subject, selected_references=["All"], max_workers=3):
    print(f"\n--- Starting Concurrent Blueprint Generation: {target_subject} ---")
    
    if not os.path.exists(blueprint_path):
        print(f"[!] Error: Blueprint file '{blueprint_path}' not found.")
        return

    with open(blueprint_path, 'r', encoding='utf-8') as f:
        blueprint = json.load(f)
    
    task_queue = []
    for requirement in blueprint:
        count = requirement.get("items_to_generate", 0)
        for _ in range(count):
            task_queue.append({
                "subject": target_subject,
                "competency": requirement["competency"],
                "bloom": requirement["bloom_level"],
                "refs": selected_references
            })
            
    total_tasks = len(task_queue)
    print(f"Total items defined by blueprint: {total_tasks}")
    print(f"Concurrent Worker Threads: {max_workers}")
    print(f"Textbook Filter Scope: {selected_references}\n")
    
    mock_exam = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                process_single_item,
                i + 1,
                total_tasks,
                t["subject"],
                t["competency"],
                t["bloom"],
                t["refs"]
            ): i for i, t in enumerate(task_queue)
        }
        
        for future in concurrent.futures.as_completed(future_to_task):
            res = future.result()
            if res:
                mock_exam.append(res)
                
    safe_subject = target_subject.replace(" ", "_").replace("-", "_").lower()
    output_filename = generate_unique_filename(f"{safe_subject}_mock_exam")
    output_path = os.path.join(os.getcwd(), output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mock_exam, f, indent=4)
        
    print(f"\n--- Blueprint Generation Complete ---")
    print(f"Successfully saved {len(mock_exam)} items to: {output_filename}")


def run_custom_generation(max_workers=3):
    print("\n--- Starting Concurrent Custom Generation ---")
    
    subject = input("Enter Subject (e.g., Industrial-Organizational Psychology): ").strip()
    competency = input("Enter Competency: ").strip()
    bloom = input("Enter Bloom's Level (e.g., Analyzing): ").strip()
    
    print("\nEnter specific textbooks separated by commas (e.g., TOS_Outline_IndusPsych_Final.pdf)")
    print("Or press Enter to query all ingested textbooks in ChromaDB.")
    pdf_input = input("Selected Textbooks: ").strip()
    
    selected_pdfs = ["All"]
    if pdf_input:
        selected_pdfs = [pdf.strip() for pdf in pdf_input.split(",") if pdf.strip()]
    
    try:
        count = int(input("How many items do you want to generate? ").strip())
    except ValueError:
        print("[!] Invalid input. Defaulting to 1 item.")
        count = 1
        
    mock_exam = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(
                process_single_item,
                i + 1,
                count,
                subject,
                competency,
                bloom,
                selected_pdfs
            ): i for i in range(count)
        }
        
        for future in concurrent.futures.as_completed(future_to_item):
            res = future.result()
            if res:
                mock_exam.append(res)
            
    output_filename = generate_unique_filename("custom_mock_exam")
    output_path = os.path.join(os.getcwd(), output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mock_exam, f, indent=4)
        
    print(f"\n--- Custom Generation Complete ---")
    print(f"Successfully saved {len(mock_exam)} items to: {output_filename}")


if __name__ == "__main__":
    print("========================================")
    print("  PSYCHOMETRIC PIPELINE CONTROL PANEL   ")
    print("========================================")
    print("1. Generate from Strict TOS Blueprint")
    print("2. Generate Custom Items (Manual Entry)")
    
    mode_choice = input("\nSelect a generation mode (1 or 2): ").strip()
    ACTIVE_WORKERS = 3
    
    if mode_choice == "1":
        print("\nAvailable Blueprints:")
        for key, data in BLUEPRINT_MAP.items():
            print(f"  {key}. {data['subject']}")
            
        bp_choice = input("Select a subject blueprint to run: ").strip()
        
        if bp_choice in BLUEPRINT_MAP:
            target_data = BLUEPRINT_MAP[bp_choice]
            
            print("\nEnter specific textbooks separated by commas (e.g., TOS_Outline_IndusPsych_Final.pdf)")
            print("Or press Enter to query all ingested textbooks in ChromaDB.")
            pdf_input = input("Selected Textbooks: ").strip()
            
            selected_pdfs = ["All"]
            if pdf_input:
                selected_pdfs = [pdf.strip() for pdf in pdf_input.split(",") if pdf.strip()]
                
            run_batch_generation(
                blueprint_path=target_data["file"], 
                target_subject=target_data["subject"],
                selected_references=selected_pdfs,
                max_workers=ACTIVE_WORKERS
            )
        else:
            print("[!] Invalid blueprint selection. Terminating.")
            
    elif mode_choice == "2":
        run_custom_generation(max_workers=ACTIVE_WORKERS)
        
    else:
        print("[!] Invalid selection. Terminating.")