# BLEPP Automated Assessment Engine (RAG Backend)

An enterprise-grade Retrieval-Augmented Generation (RAG) backend designed for the Philippine Psychometrician Licensure Examination (BLEPP). The engine parses complex psychometrics textbooks using a hybrid layout parser, indexes chunks into Supabase `pgvector`, executes two-stage dense retrieval with cross-encoder reranking, and synthesizes psychometrically sound assessment items evaluated against automated RAGAS faithfulness gates.

---

## Architecture Overview

```text
[Textbooks (PDF)] ──► [PyMuPDF (Fast Text) / Docling (Tables)] ──► [BGE-Base Embeddings] ──► [Supabase pgvector]
                                                                                                    │
[Exam Blueprints] ──► [Dynamic Query Generator] ───────────────────────────────────────────────────┤
                                                                                                    ▼
[Generated Items] ◄── [RAGAS Faithfulness Gate] ◄── [Gemini 3.5 Flash Lite] ◄── [BGE Cross-Encoder Reranker]
```

* **Document Ingestion**: Fast extraction via PyMuPDF with automated fallback to Docling for complex tabular layouts, enriched with `HybridChunker` (1,000 max tokens) and CPU thread management.
* **Vector Store**: Supabase PostgreSQL (`pgvector`) using `BAAI/bge-base-en-v1.5` normalized dense embeddings.
* **Two-Stage Retrieval**: Dense vector candidate search ($k=40$) followed by token-overlap title filtering and deep cross-encoder reranking with `BAAI/bge-reranker-v2-m3`.
* **Item Generation & Evaluation**: Chain-of-thought vignette generation via `gemini-3.5-flash-lite`, accompanied by multi-metric RAGAS evaluation (`Faithfulness`, `AnswerRelevancy`, `ContextRelevance`) and automated distractor plausibility validation.
* **Task Queue & Orchestration**: Distributed background processing via Celery and Redis, featuring thread-safe multi-API key rotation and Celery Beat worker heartbeats.

---

## Directory Layout

```text
RAG_Assessment_Engine/
├── abnormal_psychology_blueprint.json                     # Domain competency blueprint
├── developmental_psychology_blueprint.json                # Domain competency blueprint
├── industrial_organizational psychology_blueprint.json    # Domain competency blueprint
├── psychological_assessment_psychometrician_blueprint.json# Domain competency blueprint
├── celery_app.py              # Celery instance, Beat schedules, and task configurations
├── generate_exam.py           # Celery batch item generation, RAGAS gates, & DB persistence
├── hybrid_ingestion.py        # Celery ingestion task (PyMuPDF + Docling hybrid pipeline)
├── retrieve_and_rerank.py     # 2-stage retrieval (pgvector + Cross-Encoder reranker)
├── run_trial.py               # Standalone CLI batch runner for offline blueprint generation
├── main.py                    # FastAPI web server entry point
├── Dockerfile                 # Container build definition
├── docker-compose.yml         # Container orchestration (Redis, API, CPU/GPU Workers, Beat)
├── requirements.txt           # Python package dependencies
├── .env                       # Local environment variables (gitignored)
└── .gitignore                 # Git ignore rules
```

---

## Prerequisites

* **Python**: `3.10` or `3.11`
* **Redis**: Local instance or running container (port `6379`)
* **Supabase Project**: With the `pgvector` extension enabled and storage buckets configured
* **Google Gemini API Key(s)**: One or more valid Gemini API keys
* **Docker & Docker Compose**: (Optional, recommended for isolated container orchestration)

---

## Step 1: Clone & Environment Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/RAG_Assessment_Engine.git](https://github.com/YourUsername/RAG_Assessment_Engine.git)
   cd RAG_Assessment_Engine
   ```

2. **Create and activate a virtual environment:**
   * **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   * **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies:**
   * **For CPU execution (Default / No local NVIDIA GPU):**
     ```bash
     pip install --no-cache-dir torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
     pip install -r requirements.txt
     ```
   * **For GPU execution (NVIDIA CUDA 12.4+):**
     ```bash
     pip install -r requirements.txt
     pip install --force-reinstall torch torchvision --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
     ```

---

## Step 2: Configure Environment Variables

Create a `.env` file in the root project directory:

```ini
# Supabase Configuration
SUPABASE_URL=[https://your-project.supabase.co](https://your-project.supabase.co)
SUPABASE_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key

# Google Gemini API Keys (Supports comma-separated keys for automated round-robin rotation)
GEMINI_API_KEYS=AIzaSyA...,AIzaSyB...,AIzaSyC...

# Redis Broker & Celery
REDIS_HOST=localhost
REDIS_PORT=6379
CELERY_BROKER_URL=redis://localhost:6379/0

# PyTorch Execution Flags (Prevents Windows/WSL2 compile deadlocks)
TORCH_COMPILE_DISABLE=1
TORCHDYNAMO_DISABLE=1
```

---

## Step 3: Supabase Schema & Storage Setup

Ensure your Supabase project contains the following prerequisites:

1. **Storage Bucket**:
   * Create a bucket named `textbooks`.
   * Create two directory prefixes inside: `pending/` and `indexed/`.

2. **Database Schema & Vector Search**:
   * Enable the `vector` extension in the Supabase SQL Editor:
     ```sql
     CREATE EXTENSION IF NOT EXISTS vector;
     ```
   * Create the `Documents` table and the `match_documents` RPC function for cosine similarity vector matching (`768` dimensions for `BAAI/bge-base-en-v1.5`).
   * Verify supporting tables exist:
     * `Material Requests` (Tracks textbook ingestion and storage moves)
     * `Mock Exam Items` (Stores generated psychometric items and validation telemetry)
     * `Domain_Settings` (Supplies real-world organizational/clinical settings for vignette scenarios)
     * `Token_Logs` (Tracks prompt and completion token consumption)
     * `Exams` (Tracks global exam session statuses)
     * `Worker_Health` (Receives Celery Beat heartbeat updates)

---

## Step 4: Running the Engine

### Option A: Local CLI Generation Trial (No Celery/Redis Required)

To execute an end-to-end blueprint generation batch locally and export results directly to a JSON file:

1. Place your target blueprint JSON in the project root (e.g., `psychological_assessment_psychometrician_blueprint.json`).
2. Run the trial script:
   ```bash
   python run_trial.py
   ```
3. The script retrieves textbook context, generates vignettes, runs RAGAS faithfulness scoring, and outputs an artifact named `psychological_assessment_trial_exam_<timestamp>.json`.

---

### Option B: Containerized Stack via Docker Compose (Recommended)

To spin up the entire cluster (Redis, FastAPI, Celery Beat, Celery Worker, and Cloudflare Tunnel):

* **CPU Deployment:**
  ```bash
  docker compose --profile cpu up --build
  ```

* **GPU-Accelerated Deployment (Requires NVIDIA Container Toolkit):**
  ```bash
  docker compose --profile gpu up --build
  ```

To spin down the cluster:
```bash
docker compose down
```

---

## Verification & Sanity Checks

1. **Test pgvector Dense Retrieval:**
   ```bash
   python -c "from retrieve_and_rerank import retrieve_and_rerank; res = retrieve_and_rerank('item response theory theta', top_k=5, final_k=2); print('Found chunks:', len(res))"
   ```

2. **Verify Celery Worker Connectivity:**
   ```bash
   celery -A celery_app inspect ping
   ```
