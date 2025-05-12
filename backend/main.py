#!/usr/bin/env python3
"""
Ailo Forge Backend (main.py)

Features:
  • List available models
  • Modify in-memory model generation settings
  • Chat with HF Hub models, or fallback to OpenAI GPT-4
  • Fine-tune HF Hub models via Hugging Face Trainer with real checkpointing & optional Hub push
  • Pollable progress endpoint
"""
import os
import logging
import uuid
import json
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# OpenAI v1 client
from openai import OpenAI
# Transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
# HF Hub utilities
from huggingface_hub import login as hf_login, snapshot_download

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load env & init
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO)

# Optional MongoDB (unused in core flows)
if os.getenv("MONGO_URI"):
    from motor.motor_asyncio import AsyncIOMotorClient
    client_db = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    DB_NAME = os.getenv("MONGODB_DB_NAME")
    db = client_db.get_default_database() if not DB_NAME else client_db[DB_NAME]

# HF Hub auth
HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN")
HF_HUB_REPO_ID = os.getenv("HF_HUB_REPO_ID")
if HF_HUB_TOKEN:
    hf_login(HF_HUB_TOKEN)

# OpenAI client
client_openai: Optional[OpenAI] = None
if os.getenv("OPENAI_API_KEY"):
    client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    logging.warning("OPENAI_API_KEY not set; GPT-4 fallback disabled")

# Model mappings
MODEL_REPO_IDS: Dict[str, str] = {
    "7B-BASE":   "meta-llama/Llama-2-7b",
    "67B-CHAT":  "meta-llama/Llama-2-7b-chat-hf",
    "LLAMA-3-8B": "meta-llama/Llama-3-8b",
    "MISTRAL-7B": "mistralai/mistral-7b",
    "FALCON-40B": "tiiuae/falcon-40b",
    "GPT-J-6B":   "EleutherAI/gpt-j-6B",
    "DEEPSEEK-CODER-33B": "deepseek/coder-33b",
}

# In-memory stores
config_store: Dict[str, Dict[str, Any]] = {}
progress_store: Dict[str, Dict[str, Any]] = {}
model_pipelines: Dict[str, Any] = {}

# Create app
app = FastAPI(title="Ailo Forge", version="1.0.0", debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# Schemas
class ModifyRequest(BaseModel):
    model_version: str = Field(..., alias="modelVersion")
    temperature: float
    token_limit: int = Field(..., alias="tokenLimit")
    instructions: str = Field("", alias="instructions")

    class Config:
        allow_population_by_alias = True
        populate_by_name = True

class RunRequest(BaseModel):
    model_version: str = Field(..., alias="modelVersion")
    prompt: str

    class Config:
        allow_population_by_alias = True
        populate_by_name = True

# Healthcheck
@app.get("/")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok", "message": "Ailo Forge backend live"}

# List models
@app.get("/models")
async def list_models() -> Dict[str, List[str]]:
    return {"models": list(MODEL_REPO_IDS.keys())}

# Modify config in-memory
@app.post("/modify-file")
async def modify_file(req: ModifyRequest) -> Dict[str, Any]:
    if req.model_version not in MODEL_REPO_IDS:
        raise HTTPException(404, detail="Unknown model version")
    config_store[req.model_version] = {
        "temperature": req.temperature,
        "token_limit": req.token_limit,
        "instructions": req.instructions,
    }
    model_pipelines.pop(req.model_version, None)
    return {"success": True, "message": "Config updated in-memory"}

# Run chat/fallback
@app.post("/run")
async def run_model(req: RunRequest) -> Dict[str, Any]:
    mv, prompt = req.model_version, req.prompt
    if mv not in MODEL_REPO_IDS and not client_openai:
        raise HTTPException(404, detail="Model not available")
    cfg = config_store.get(mv, {})
    temp = cfg.get("temperature", 0.7)
    max_len = cfg.get("token_limit", 150)
    instr = cfg.get("instructions", "")

    repo_id = MODEL_REPO_IDS.get(mv)
    if repo_id:
        src = snapshot_download(repo_id, cache_dir="models")
        if mv not in model_pipelines:
            tok = AutoTokenizer.from_pretrained(src)
            mod = AutoModelForCausalLM.from_pretrained(src)
            model_pipelines[mv] = pipeline("text-generation", model=mod, tokenizer=tok, device_map="auto")
        full_prompt = f"{instr}\n{prompt}".strip()
        out = model_pipelines[mv](full_prompt, max_length=max_len, temperature=temp)
        text = out[0]["generated_text"]
    else:
        resp = client_openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role":"system","content":instr}, {"role":"user","content":prompt}],
            temperature=temp,
            max_tokens=max_len,
        )
        text = resp.choices[0].message.content.strip()

    return {"success": True, "response": text}

# Train endpoint
@app.post("/train")
async def train_model(
    base_model: str = Form(...), trainingObjective: str = Form(...),
    files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    if base_model not in MODEL_REPO_IDS:
        raise HTTPException(404, detail="Unknown model version")
    texts = []
    for f in files:
        data = await f.read()
        if not data.startswith((b"\xFF\xD8", b"\x89PNG")):
            texts.append(data.decode(errors="ignore"))
    if not texts:
        raise HTTPException(400, detail="No valid text data provided")
    job_id = str(uuid.uuid4())
    progress_store[job_id] = {"percent": 0, "status": "in_progress"}
    background_tasks.add_task(_run_training, job_id, base_model, texts, trainingObjective)
    return {"job_id": job_id, "status": "training_started"}

# Background training

def _run_training(job_id: str, base_model: str, texts: List[str], objective: str) -> None:
    try:
        repo_id = MODEL_REPO_IDS[base_model]
        src = snapshot_download(repo_id, cache_dir="models")
        tok = AutoTokenizer.from_pretrained(src)
        mod = AutoModelForCausalLM.from_pretrained(src)
        ds = Dataset.from_dict({"text": texts})
        ds = ds.map(lambda x: tok(x["text"], truncation=True, max_length=128), batched=True)
        out_dir = os.path.join("models", f"{base_model}-ft-{job_id}")
        args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            logging_steps=10,
            save_steps=50,
            push_to_hub=bool(HF_HUB_TOKEN),
            hub_token=HF_HUB_TOKEN,
            hub_model_id=HF_HUB_REPO_ID or f"{base_model}-ft-{job_id}",
        )
        trainer = Trainer(model=mod, args=args, train_dataset=ds, tokenizer=tok)
        trainer.train()
        trainer.save_model(out_dir)
        if HF_HUB_TOKEN:
            trainer.push_to_hub()
        progress_store[job_id] = {"percent": 100, "status": "completed"}
    except Exception:
        logging.exception("Training failed")
        progress_store[job_id] = {"percent": 0, "status": "failed"}

# Progress
@app.get("/progress/{job_id}")
async def get_progress(job_id: str) -> Dict[str, Any]:
    if job_id in progress_store:
        return progress_store[job_id]
    raise HTTPException(404, detail="Job not found")

# Run
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
