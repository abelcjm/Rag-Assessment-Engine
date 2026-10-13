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

# Safe import handling for both filename conventions
try:
    from hybrid_ingestion import batch_ingest_task
except ImportError:
    from hybrid_ingestion import batch_ingest_task

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
    custom_settings: Optional[List[str]] = None

class CustomBlock(BaseModel):
    competency: str
    bloom: str
    count: int
    setting: Optional[str] = None                   # Single setting from frontend
    custom_settings: Optional[List[str]] = None     # Array of settings if provided

class CustomBatchRequest(BaseModel):
    subject: str
    blocks: List[CustomBlock]
    references: List[str] = ["All"]
    exam_session_id: str
    custom_settings: Optional[List[str]] = None

class RegenerateRequest(BaseModel):
    old_item_id: str
    exam_session_id: str


BLUEPRINT_MAP = {
    "1": {"subject": "Abnormal Psychology", "file": "abnormal_psychology_blueprint.json"},
    "2": {"subject": "Industrial-Organizational Psychology", "file": "industrial_organizational psychology_blueprint.json"},
    "3": {"subject": "Psychological Assessment", "file": "psychological_assessment_psychometrician_blueprint.json"},
    "4": {"subject": "Developmental Psychology", "file": "developmental_psychology_blueprint.json"}
}


@app.get("/health")
@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok", "service": "Psychometric RAG Assessment Engine"}


# --- Vectorization & Document Management Endpoints ---
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
        file_name = os.path.basename(request.storage_path)

        # Delete all document vectors matching storage_path or filename within metadata JSONB
        response = (
            supabase.table("Documents")
            .delete()
            .or_(f"metadata->>storage_path.eq.{request.storage_path},metadata->>source.eq.{file_name}")
            .execute()
        )
        
        deleted_count = len(response.data) if response.data else 0
        print(f"[VECTOR PURGE] Successfully removed {deleted_count} chunks for {file_name}")
        return {
            "message": f"Successfully deleted {deleted_count} vectors for {file_name}",
            "deleted_chunks": deleted_count
        }
    except Exception as e:
        print(f"[VECTOR PURGE ERROR]: {str(e)}")
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


# --- Exam Generation Endpoints ---
@app.post("/api/v1/generate/blueprint")
async def generate_blueprint_async(request: BlueprintGenerationRequest):
    blueprint_config = BLUEPRINT_MAP.get(str(request.blueprint_id))
    if not blueprint_config or not os.path.exists(blueprint_config["file"]):
        raise HTTPException(status_code=404, detail="Invalid blueprint ID or blueprint file missing.")

    with open(blueprint_config["file"], 'r', encoding='utf-8') as f:
        blueprint_data = json.load(f)

    tasks = []
    for requirement in blueprint_data:
        count = requirement.get("items_to_generate", 0)
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
            # Resolve setting: priority is block-level single setting -> block-level list -> global settings
            if block.setting:
                active_settings = [block.setting]
            elif block.custom_settings:
                active_settings = block.custom_settings
            else:
                active_settings = request.custom_settings
            
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
        raw_refs = exam_resp.data[0].get("references") if exam_resp.data else None
        references = raw_refs if isinstance(raw_refs, list) and len(raw_refs) > 0 else ["All"]
        
        # Dispatch Celery task targeting the exact row ID for in-place replacement
        task = generate_assessment_batch_task.delay(
            subject=old_item.get("subject") or "General Scope",
            competency=old_item.get("competency") or "General Competency",
            target_blooms_level=old_item.get("blooms_level") or "Understanding",
            items_to_generate=1,
            selected_references=references,
            exam_session_id=request.exam_session_id,
            custom_settings=None,
            replace_item_id=request.old_item_id
        )
        return {"message": "Regeneration task queued successfully.", "task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))