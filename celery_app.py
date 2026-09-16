from celery import Celery

# Initialize Celery and point it to the local Redis Docker container
celery_app = Celery(
    "psychometric_engine",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
    include=["hybrid_ingestion", "generate_exam"] 
)

# Optional: Configure routing and concurrency limits
celery_app.conf.update(
    task_track_started=True,
    result_extended=True,
    worker_prefetch_multiplier=1 # Prevents one worker from hoarding all tasks
)