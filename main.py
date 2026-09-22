import json
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from celery.result import AsyncResult

from celery_app import celery_app
from hybrid_ingestion import batch_ingest_task
from generate_exam import generate_assessment_item_task

app = FastAPI(title="Psychometric RAG Assessment Engine")

# Configure CORS to allow requests from local development and Vercel deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://rag-assessment-prototype.vercel.app"
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class IngestionRequest(BaseModel):
    file_path: str  

BLUEPRINT_MAP = {
    "1": {"subject": "Abnormal Psychology", "file": "abnormal_psychology_blueprint.json"},
    "2": {"subject": "Industrial-Organizational Psychology", "file": "industrial_organizational psychology_blueprint.json"},
    "3": {"subject": "Psychological Assessment", "file": "psychological_assessment_psychometrician_blueprint.json"},
    "4": {"subject": "Developmental Psychology", "file": "developmental_psychology_blueprint.json"}
}

@app.post("/api/ingest")
async def trigger_ingestion(request: IngestionRequest):
    task = batch_ingest_task.delay(request.file_path)
    return {"message": f"Ingestion started for {request.file_path}", "task_id": task.id}

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

@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "status": task_result.status}
    
    if task_result.status == 'PROCESSING':
        response["details"] = task_result.info 
    elif task_result.status == 'SUCCESS':
        response["details"] = {"message": "Item generated and saved to database."}
        response["final_result"] = task_result.result
    elif task_result.status == 'FAILURE':
        response["details"] = str(task_result.info)
        
    return response