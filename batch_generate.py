import json
import os
import time
from dotenv import load_dotenv
from generate_exam import generate_assessment_item

load_dotenv()

def load_blueprint(filepath):
    """
    Loads and parses the Table of Specifications (TOS) examination blueprint.

    Args:
        filepath (str): Path to the blueprint JSON file.

    Returns:
        list[dict]: Array of competency requirements and generation targets.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_batch_generation(blueprint_path, output_filename, selected_references=["All"]):
    """
    Iterates through blueprint specifications to generate and validate exam items in batch.

    Saves the aggregated results to an output JSON file.

    Args:
        blueprint_path (str): Filepath to the TOS blueprint JSON.
        output_filename (str): Target filename for generated examination output.
        selected_references (list[str]): Document filter list passed to retrieval.
    """
    print(f"--- Starting Batch Generation: {blueprint_path} ---")
    blueprint = load_blueprint(blueprint_path)
    
    mock_exam = []
    total_items_to_generate = sum(req.get("items_to_generate", 0) for req in blueprint)
    items_generated = 0
    
    print(f"Total items required by TOS: {total_items_to_generate}")
    
    for requirement in blueprint:
        subject = requirement["subject"]
        competency = requirement["competency"]
        bloom_level = requirement["bloom_level"]
        count = requirement["items_to_generate"]
        
        print(f"\nProcessing Competency: {competency}")
        print(f"Target: {count} item(s) at '{bloom_level}' level.")
        
        for i in range(count):
            print(f"  -> Generating item {i+1} of {count}...")
            
            try:
                item = generate_assessment_item(
                    subject=subject,
                    competency=competency,
                    target_blooms_level=bloom_level,
                    selected_references=selected_references
                )
                
                # Normalize Pydantic models or dict outputs to a standard Python dictionary
                if hasattr(item, 'model_dump'):
                    item_dict = item.model_dump()
                elif hasattr(item, 'dict'):
                    item_dict = item.dict()
                else:
                    item_dict = dict(item)
                
                mock_exam.append(item_dict)
                items_generated += 1
                
                # Rate limit pacing to avoid Gemini API quota exhaustions
                time.sleep(2)
                
            except Exception as e:
                print(f"  [!] Error generating item: {e}")
                
    output_path = os.path.join(os.getcwd(), output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mock_exam, f, indent=4)
        
    print(f"\n--- Batch Generation Complete ---")
    print(f"Successfully generated {items_generated} / {total_items_to_generate} items.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    blueprint_file = "abnormal_psych_blueprint.json"
    output_file = "abnormal_psych_mock_exam.json"
    
    if os.path.exists(blueprint_file):
        run_batch_generation(blueprint_file, output_file)
    else:
        print(f"Error: {blueprint_file} not found.")