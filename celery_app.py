import os
from celery import Celery

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

# Initialize Celery and point it to the dynamic Redis URL
celery_app = Celery(
    "psychometric_engine",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["hybrid_ingestion", "generate_exam"] 
)

# Configure routing, concurrency limits, tracking, and UPGRADED time limits
celery_app.conf.update(
    task_track_started=True,
    result_extended=True,
    
    # Prevents one worker from hoarding tasks it doesn't have RAM for
    worker_prefetch_multiplier=1, 
    
    # --- NEW BATCH GENERATION TIMEOUT UPGRADES ---
    # Increased because a single task now generates up to 4 CoT items, 
    # executes query rewrites, and runs RAGAS on the batch.
    task_soft_time_limit=120,   # 2 minutes before raising SoftTimeLimitExceeded
    task_time_limit=150,        # 2.5 minutes hard kill to prevent deadlocks
    
    # CRITICAL: Ensures tasks are re-queued if the worker crashes (e.g., GPU Out of Memory)
    task_acks_late=True         
)

# --- CELERY BEAT SCHEDULE FOR NEXT.JS DASHBOARD ---
celery_app.conf.beat_schedule = {
    'worker-health-heartbeat': {
        'task': 'generate_exam.heartbeat_task',
        'schedule': 60.0,
    },
}