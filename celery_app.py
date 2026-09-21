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

# Optional: Configure routing and concurrency limits
celery_app.conf.update(
    task_track_started=True,
    result_extended=True,
    worker_prefetch_multiplier=1 # Prevents one worker from hoarding all tasks
)