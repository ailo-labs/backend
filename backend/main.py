#!/usr/bin/env python3
"""
Ailo Forge Backend (main.py)

Features:
  • Modify in-memory chat settings (temperature, tokens, instructions)
  • Chat via Hugging Face Text-Generation-Inference endpoint or fallback to OpenAI GPT-4
  • Fine-tune Hugging Face Hub models via Trainer with optional push to Hub
  • Pollable training progress endpoint
"""
import os
import logging
import uuid
import json
from typing import List, Dict, Any

import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import openai
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

# ─────────────────────────────────────────────────────────────────────────────
# 1) Environment & Initialization
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO)

# OpenAI GPT-4 setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required for chat endpoint")
openai.api_key = OPENAI_API_KEY
logging.info("Configured OpenAI GPT-4 client")

# Hugging Face TGI endpoint (optional)
HF_INFERENCE_URL = os.getenv("HF_INFERENCE_URL")  # e.g. http://localhost:8080/v1
if HF_INFERENCE_URL:
    logging.info(f"Using Hugging Face TGI at {HF_INFERENCE_URL}")

# Hugging Face Hub token (for training push)
HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN")

# In-memory stores
aido_config: Dict[str, Any] = {}
train_progress: Dict[str, Dict[str, Any]] = {}

# FastAPI setup
app = FastAPI(title="Ailo Forge", version="3.0.0", debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# ─────────────────────────────────────────────────────────────────────────────
# 2) Schemas
# ─────────────────────────────────────────────────────────────────────────────
class ModifyChat(BaseModel):
    temperature: float
    token_limit: int = Field(..., alias="tokenLimit")
    instructions: str = Field("", alias="instructions")
    class Config:
        allow_population_by_alias = True
        allow_population_by_field_name = True

class RunChat(BaseModel):
    prompt: str

# ─────────────────────────────────────────────────────────────────────────────
# 3) Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def healthcheck():
    return {"status": "ok", "message": "Ailo Forge live"}

@app.post("/modify-chat")
async def modify_chat(req: ModifyChat):
    # Save chat settings under key 'gpt4'
    aido_config['gpt4'] = {
        "temperature": req.temperature,
        "max_tokens": req.token_limit,
        "instructions": req.instructions,
    }
    return {"success": True, "message": "Chat config updated"}

@app.post("/run")
async def run_chat(req: RunChat):
    cfg = aido_config.get('gpt4', {})
    temperature = cfg.get('temperature', 0.7)
    max_tokens = cfg.get('max_tokens', 150)
    instructions = cfg.get('instructions', "")

    messages = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    messages.append({"role": "user", "content": req.prompt})

    # 1) Try HF TGI if configured
    if HF_INFERENCE_URL:
        payload = {
            "model": "tgi",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            resp = requests.post(f"{HF_INFERENCE_URL}/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            # TGI returns choices like OpenAI
            content = data['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"Hugging Face Inference error: {e}")
            # fallback to OpenAI
        else:
            return {"success": True, "response": content}

    # 2) Fallback to OpenAI GPT-4
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI error: {e}")
        raise HTTPException(502, detail=str(e))

    return {"success": True, "response": content}

@app.post("/train")
async def train_model(
    repo_id: str = Form(...),
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    # Collect text data
    texts: List[str] = []
    for f in files:
        data = await f.read()
        if not data.startswith((b"\xFF\xD8", b"\x89PNG")):
            texts.append(data.decode("utf-8", errors="ignore"))
    if not texts:
        raise HTTPException(400, detail="No valid text data provided")

    job_id = str(uuid.uuid4())
    train_progress[job_id] = {"percent": 0, "status": "in_progress"}
    background_tasks.add_task(_run_training, job_id, repo_id, texts)
    return {"job_id": job_id, "status": "training_started"}

@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    if job_id not in train_progress:
        raise HTTPException(404, detail="Job not found")
    return train_progress[job_id]

# ─────────────────────────────────────────────────────────────────────────────
# 4) Background Training
# ─────────────────────────────────────────────────────────────────────────────

def _run_training(job_id: str, repo_id: str, texts: List[str]):
    try:
        tok = AutoTokenizer.from_pretrained(repo_id)
        mod = AutoModelForCausalLM.from_pretrained(repo_id)
        ds = Dataset.from_dict({"text": texts})
        ds = ds.map(lambda x: tok(x['text'], truncation=True, max_length=128), batched=True)
        out_dir = f"models/ft-{job_id}"
        args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            push_to_hub=bool(HF_HUB_TOKEN),
            hub_token=HF_HUB_TOKEN,
            hub_model_id=os.getenv("HF_HUB_REPO_ID", None) or f"ft-{job_id}",
        )
        trainer = Trainer(model=mod, args=args, train_dataset=ds, tokenizer=tok)
        trainer.train()
        trainer.save_model(out_dir)
        if HF_HUB_TOKEN:
            trainer.push_to_hub()
        train_progress[job_id] = {"percent": 100, "status": "completed"}
    except Exception:
        logging.exception("Training failed")
        train_progress[job_id] = {"percent": 0, "status": "failed"}

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
