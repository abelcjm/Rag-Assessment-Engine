import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from celery.result import AsyncResult
from celery import chord
from supabase import create_client, Client

from celery_app import celery_app
from hybrid_ingestion import batch_ingest_task

# UPGRADED: Importing the new batch task
from generate_exam import (
    generate_assessment_batch_task,
    finalize_exam_session_task,
    fail_exam_session_task
)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

app = FastAPI(title="Psychometric RAG Assessment Engine")

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

# --- Pydantic Models ---
class VectorizeRequest(BaseModel):
    request_id: str
    storage_path: str
    tags: List[str] = []

class DeleteVectorsRequest(BaseModel):
    storage_path: str

class BlueprintGenerationRequest(BaseModel):
    blueprint_id: str
    references: List[str] = ["All"]
    exam_session_id: str
    custom_settings: Optional[List[str]] = None  # NEW: Global settings override

class CustomBlock(BaseModel):
    competency: str
    bloom: str
    count: int
    custom_settings: Optional[List[str]] = None  # NEW: Block-level settings override

class CustomBatchRequest(BaseModel):
    subject: str
    blocks: List[CustomBlock]
    references: List[str] = ["All"]
    exam_session_id: str
    custom_settings: Optional[List[str]] = None  # NEW: Global settings override

class RegenerateRequest(BaseModel):
    old_item_id: str
    exam_session_id: str


BLUEPRINT_MAP = {
    "1": {"subject": "Abnormal Psychology", "file": "abnormal_psychology_blueprint.json"},
    "2": {"subject": "Industrial-Organizational Psychology", "file": "industrial_organizational psychology_blueprint.json"},
    "3": {"subject": "Psychological Assessment", "file": "psychological_assessment_psychometrician_blueprint.json"},
    "4": {"subject": "Developmental Psychology", "file": "developmental_psychology_blueprint.json"}
}


# --- Vectorization Endpoints ---
@app.post("/api/v1/vectorize")
async def trigger_vectorize(request: VectorizeRequest):
    task = batch_ingest_task.delay(
        request_id=request.request_id,
        storage_path=request.storage_path,
        tags=request.tags
    )
    return {"message": f"Vectorization started for {request.storage_path}", "task_id": task.id}

@app.delete("/api/v1/vectors")
async def delete_vectors(request: DeleteVectorsRequest):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not initialized.")
    try:
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


# --- Generation Endpoints using Chords ---
@app.post("/api/v1/generate/blueprint")
async def generate_blueprint_async(request: BlueprintGenerationRequest):
    blueprint_config = BLUEPRINT_MAP.get(str(request.blueprint_id))
    if not blueprint_config or not os.path.exists(blueprint_config["file"]):
        raise HTTPException(status_code=404, detail="Invalid blueprint ID.")

    with open(blueprint_config["file"], 'r', encoding='utf-8') as f:
        blueprint_data = json.load(f)

    tasks = []
    for requirement in blueprint_data:
        count = requirement.get("items_to_generate", 0)
        
        # UPGRADED: No longer looping to create single-item tasks. 
        # Fires one batch task per competency with items_to_generate passed directly.
        if count > 0:
            tasks.append(generate_assessment_batch_task.s(
                subject=blueprint_config["subject"],
                competency=requirement["competency"],
                target_blooms_level=requirement["bloom_level"],
                items_to_generate=count,
                selected_references=request.references,
                exam_session_id=request.exam_session_id,
                custom_settings=request.custom_settings
            ))

    if not tasks:
        raise HTTPException(status_code=400, detail="Blueprint generated 0 tasks.")

    callback = finalize_exam_session_task.s(request.exam_session_id)
    error_callback = fail_exam_session_task.s(request.exam_session_id)
    chord(tasks)(callback.on_error(error_callback))
    
    return {"status": "queued", "message": "Parallel blueprint batch dispatched."}

@app.post("/api/v1/generate/custom_batch")
async def generate_custom_batch_async(request: CustomBatchRequest):
    tasks = []
    for block in request.blocks:
        if block.count > 0:
            # Prioritize block-level custom settings over global custom settings
            active_settings = block.custom_settings or request.custom_settings
            
            tasks.append(generate_assessment_batch_task.s(
                subject=request.subject,
                competency=block.competency,
                target_blooms_level=block.bloom,
                items_to_generate=block.count,
                selected_references=request.references,
                exam_session_id=request.exam_session_id,
                custom_settings=active_settings
            ))
            
    if not tasks:
        raise HTTPException(status_code=400, detail="Custom batch generated 0 tasks.")

    callback = finalize_exam_session_task.s(request.exam_session_id)
    error_callback = fail_exam_session_task.s(request.exam_session_id)
    chord(tasks)(callback.on_error(error_callback))
    
    return {"status": "queued", "message": "Parallel custom batch dispatched."}

@app.post("/api/v1/generate/regenerate")
async def regenerate_item(request: RegenerateRequest):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not initialized.")
    try:
        item_resp = supabase.table("Mock Exam Items").select("subject, competency, blooms_level").eq("id", request.old_item_id).execute()
        if not item_resp.data:
            raise HTTPException(status_code=404, detail="Original item not found in the database.")
        old_item = item_resp.data[0]
        
        exam_resp = supabase.table("Exams").select("references").eq("exam_id", request.exam_session_id).execute()
        references = exam_resp.data[0].get("references", ["All"]) if exam_resp.data else ["All"]
        
        supabase.table("Mock Exam Items").update({"status": "ARCHIVED"}).eq("id", request.old_item_id).execute()
        
        # UPGRADED: Uses the batch task but with items_to_generate=1 to regenerate a single question
        task = generate_assessment_batch_task.delay(
            subject=old_item.get("subject"),
            competency=old_item.get("competency"),
            target_blooms_level=old_item.get("blooms_level"),
            items_to_generate=1,
            selected_references=references,
            exam_session_id=request.exam_session_id,
            custom_settings=None
        )
        return {"message": "Regeneration task queued successfully.", "task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))