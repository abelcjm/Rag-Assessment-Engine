import json
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from celery.result import AsyncResult
from supabase import create_client, Client

from celery_app import celery_app
from hybrid_ingestion import batch_ingest_task
from generate_exam import generate_assessment_item_task

# Initialize Supabase Client for the direct DELETE operation
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

app = FastAPI(title="Psychometric RAG Assessment Engine")

# Configure CORS to allow requests from local development and Vercel deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://rag-assessment-prototype.vercel.app"
    ],
    allow_origin_regex=r"^https://.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- New Pydantic Models ---
class VectorizeRequest(BaseModel):
    request_id: str
    storage_path: str
    tags: List[str] = []

class DeleteVectorsRequest(BaseModel):
    storage_path: str

# --- Existing Models ---
class CustomGenerationRequest(BaseModel):
    subject: str
    competency: str
    bloom: str
    count: int = 1
    selected_pdfs: List[str] = ["All"]
    exam_session_id: Optional[str] = None

class BlueprintGenerationRequest(BaseModel):
    blueprint_id: str
    selected_pdfs: List[str] = ["All"]
    exam_session_id: str

BLUEPRINT_MAP = {
    "1": {"subject": "Abnormal Psychology", "file": "abnormal_psychology_blueprint.json"},
    "2": {"subject": "Industrial-Organizational Psychology", "file": "industrial_organizational psychology_blueprint.json"},
    "3": {"subject": "Psychological Assessment", "file": "psychological_assessment_psychometrician_blueprint.json"},
    "4": {"subject": "Developmental Psychology", "file": "developmental_psychology_blueprint.json"}
}

# --- Updated Endpoints ---
@app.post("/api/v1/vectorize")
async def trigger_vectorize(request: VectorizeRequest):
    # Passes the new exact arguments to the Celery worker
    task = batch_ingest_task.delay(
        request_id=request.request_id,
        storage_path=request.storage_path,
        tags=request.tags
    )
    return {"message": f"Vectorization started for {request.storage_path}", "task_id": task.id}

@app.delete("/api/v1/vectors")
async def delete_vectors(request: DeleteVectorsRequest):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not initialized on the backend.")
    try:
        # Executes the hard purge of the vector embeddings based on the indexed path
        response = supabase.table("Documents").delete().eq("storage_path", request.storage_path).execute()
        return {"message": f"Successfully deleted vectors for {request.storage_path}", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/task/{task_id}")
async def get_task_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "status": task_result.status}
    
    if task_result.status == 'PROCESSING':
        response["details"] = task_result.info 
    elif task_result.status == 'SUCCESS':
        response["details"] = {"message": "Task successfully completed."}
        response["final_result"] = task_result.result
    elif task_result.status == 'FAILURE':
        response["details"] = str(task_result.info)
        
    return response

# --- Generation Endpoints (Unchanged) ---
@app.post("/api/generate/custom")
async def generate_custom_items(request: CustomGenerationRequest):
    task_ids = []
    for _ in range(request.count):
        task = generate_assessment_item_task.delay(
            subject=request.subject,
            competency=request.competency,
            target_blooms_level=request.bloom,
            selected_references=request.selected_pdfs,
            exam_session_id=request.exam_session_id
        )
        task_ids.append(task.id)
    return {"message": f"Queued {request.count} item(s).", "task_ids": task_ids}

@app.post("/api/generate/blueprint")
async def generate_blueprint(request: BlueprintGenerationRequest):
    blueprint_config = BLUEPRINT_MAP.get(request.blueprint_id)
    if not blueprint_config:
        raise HTTPException(status_code=404, detail="Blueprint ID not found.")
        
    blueprint_path = blueprint_config["file"]
    if not os.path.exists(blueprint_path):
         raise HTTPException(status_code=404, detail=f"Blueprint file '{blueprint_path}' missing.")

    with open(blueprint_path, 'r', encoding='utf-8') as f:
        blueprint_data = json.load(f)
        
    task_ids = []
    for requirement in blueprint_data:
        count = requirement.get("items_to_generate", 0)
        for _ in range(count):
            task = generate_assessment_item_task.delay(
                subject=blueprint_config["subject"],
                competency=requirement["competency"],
                target_blooms_level=requirement["bloom_level"],
                selected_references=request.selected_pdfs,
                exam_session_id=request.exam_session_id
            )
            task_ids.append(task.id)
            
    return {"message": f"Queued {len(task_ids)} items.", "task_ids": task_ids}