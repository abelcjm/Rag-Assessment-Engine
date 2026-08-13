```markdown
# RAG Assessment Engine - Setup & Replication Guide

This repository contains the prototype Retrieval-Augmented Generation (RAG) backend designed for local document ingestion, vector storage, neural retrieval, and question generation. Follow this guide to replicate the environment, process academic literature, and run proof-of-concept generations.

---

## Prerequisites

Before starting, ensure your local system meets the following requirements:
* **Python**: Version 3.10 or higher
* **Git**: Installed on your local machine
* **API Key**: A valid Google Gemini API Key
* **Operating System**: Windows / Linux / macOS (Windows-specific settings are noted below)

---

## Project Structure

Ensure your project root directory matches the following layout before execution:

```text
RAG_Assessment_Engine/
├── chroma_db_store/       (Created automatically during ingestion)
├── textbooks/             (Folder containing reference PDFs and TOS)
├── .env                   (Environment variables file)
├── .gitignore
├── hybrid_ingestion.py    (Hybrid parser script using PyMuPDF and Docling)
├── retrieve_and_rerank.py (Cosine similarity + BGE cross-encoder script)
├── generate_exam.py       (Pentagram prompting generation script)
└── requirements.txt

```

---

## Step 1: Environment Setup

1. **Clone the Repository**
```cmd
git clone [https://github.com/YourUsername/rag-assessment-engine.git](https://github.com/YourUsername/rag-assessment-engine.git)
cd rag-assessment-engine

```


2. **Create and Activate a Virtual Environment**
* Windows (Command Prompt):
```cmd
python -m venv venv
venv\Scripts\activate

```


* macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate

```




3. **Install Dependencies**
```cmd
pip install -r requirements.txt

```



---

## Step 2: Configure Environment Variables

Create a file named `.env` in the root directory to store your API keys and PyTorch environment flags.

Add the following contents to `.env`:

```env
# Gemini API Configuration
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Force PyTorch to use standard execution on Windows (prevents missing C++ compiler errors)
TORCH_COMPILE_DISABLE=1
TORCHDYNAMO_DISABLE=1

```

---

## Step 3: Data Ingestion and Database Creation

This phase processes reference materials using a hybrid approach (PyMuPDF for fast plain text, Docling for complex matrices) and builds the local vector database.

1. **Prepare Source Files**
Create a folder named `textbooks` in the root directory if it does not exist:
```cmd
mkdir textbooks

```


Place your Table of Specifications (TOS) PDF and reference textbook PDFs inside the `textbooks/` directory.
2. **Run the Hybrid Ingestion Script**
```cmd
python hybrid_ingestion.py

```


* **Expected Outcome**: The script scans each PDF, routes complex layout pages to Docling, chunks the content into semantic segments, converts them to vectors using `nomic-embed-text-v1.5`, and populates the local `chroma_db_store/` database.



---

## Step 4: Verify Neural Retrieval and Reranking

To confirm that the local vector store accurately fetches and re-ranks context without calling the generative AI model:

1. **Run the Retrieval Script**
```cmd
python retrieve_and_rerank.py

```


* **Expected Outcome**: The script performs a Cosine Similarity search, passes the candidate chunks to the `BAAI/bge-reranker-v2-m3` cross-encoder, and outputs the top-ranked context chunks directly in the terminal.



---

## Step 5: Test Item Generation

To synthesize a structured multiple-choice assessment item using the retrieved curriculum data:

1. **Run the Generation Script**
```cmd
python generate_exam.py

```


* **Expected Outcome**: The system passes the top-reranked textbook context into `gemini-3.6-flash` under the Pentagram Prompting Framework constraints. It outputs a validated JSON object containing a question stem, plausible distractors, the correct answer, and a rationale.
