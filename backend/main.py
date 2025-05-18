#!/usr/bin/env python3
"""
Ailo Forge Backend (main.py)

Features:
  • Modify in-memory GPT-4 generation settings
  • Chat exclusively via OpenAI GPT-4
  • Fine-tune Hugging Face models via Hugging Face Trainer
  • Pollable fine-tune progress endpoint
"""
import os
import logging
import uuid
import json
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# OpenAI
import openai
# HF Trainer
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load env & init
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO)

# OpenAI setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required for chat endpoint")
openai.api_key = OPENAI_API_KEY
logging.info("Configured OpenAI GPT-4 client")

# In-memory stores
config_store: Dict[str, Any] = {}
progress_store: Dict[str, Dict[str, Any]] = {}

# FastAPI setup
app = FastAPI(title="Ailo Forge", version="2.1.0", debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# Schemas\class ModifyRequest(BaseModel):
    temperature: float
    max_tokens: int = Field(..., alias="tokenLimit")
    instructions: str = Field("", alias="instructions")
    class Config:
        allow_population_by_alias = True
        allow_population_by_field_name = True

class RunRequest(BaseModel):
    prompt: str

# Endpoints
@app.get("/")
async def healthcheck():
    return {"status": "ok", "message": "Ailo Forge live (OpenAI chat only)"}

@app.post("/modify-chat")
async def modify_chat(req: ModifyRequest):
    # store settings under key 'gpt4'
    config_store['gpt4'] = {
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "instructions": req.instructions,
    }
    return {"success": True, "message": "Chat config updated"}

@app.post("/run")
async def run_chat(req: RunRequest):
    cfg = config_store.get('gpt4', {})
    temperature = cfg.get('temperature', 0.7)
    max_tokens = cfg.get('max_tokens', 150)
    instructions = cfg.get('instructions', "")

    messages = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    messages.append({"role": "user", "content": req.prompt})

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logging.error(f"OpenAI error: {e}")
        raise HTTPException(502, detail=str(e))

    content = resp.choices[0].message.content.strip()
    return {"success": True, "response": content}

@app.post("/train")
async def train_model(
    repo_id: str = Form(...),
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    # collect text
    texts: List[str] = []
    for f in files:
        data = await f.read()
        if not data.startswith((b"\xFF\xD8", b"\x89PNG")):
            texts.append(data.decode("utf-8", errors="ignore"))
    if not texts:
        raise HTTPException(400, detail="No valid text data provided")

    job_id = str(uuid.uuid4())
    progress_store[job_id] = {"percent": 0, "status": "in_progress"}
    background_tasks.add_task(_run_training, job_id, repo_id, texts)
    return {"job_id": job_id, "status": "training_started"}

def _run_training(job_id: str, repo_id: str, texts: List[str]):
    try:
        tok = AutoTokenizer.from_pretrained(repo_id)
        mod = AutoModelForCausalLM.from_pretrained(repo_id)
        ds = Dataset.from_dict({"text": texts})
        ds = ds.map(lambda x: tok(x['text'], truncation=True, max_length=128), batched=True)
        out_dir = f"models/{job_id}"
        args = TrainingArguments(output_dir=out_dir, num_train_epochs=3, per_device_train_batch_size=2)
        trainer = Trainer(model=mod, args=args, train_dataset=ds, tokenizer=tok)
        trainer.train()
        trainer.save_model(out_dir)
        progress_store[job_id] = {"percent": 100, "status": "completed"}
    except Exception:
        logging.exception("Training failed")
        progress_store[job_id] = {"percent": 0, "status": "failed"}

@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    if job_id not in progress_store:
        raise HTTPException(404, detail="Job not found")
    return progress_store[job_id]

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
