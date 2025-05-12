#!/usr/bin/env python3
"""
Ailo Forge Backend (main.py)

Features:
  • List available models
  • Modify model generation settings in-place
  • Chat with local HF models, or fallback to OpenAI GPT-4
  • Fine-tune base models via Hugging Face Trainer with real checkpointing & optional Hub push
  • Pollable progress endpoint
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
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

import openai
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from huggingface_hub import login as hf_login, snapshot_download

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load env & initialize
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO)

# OpenAI GPT-4 fallback
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    logging.warning("OPENAI_API_KEY not set; GPT-4 fallback disabled")

# Hugging Face Hub
HF_HUB_TOKEN   = os.getenv("HF_HUB_TOKEN")
HF_HUB_REPO_ID = os.getenv("HF_HUB_REPO_ID")
if HF_HUB_TOKEN:
    hf_login(HF_HUB_TOKEN)

# MongoDB (state storage)
MONGO_URI = os.getenv("MONGO_URI") or ""
if not MONGO_URI:
    raise RuntimeError("MONGO_URI is required")
client = AsyncIOMotorClient(MONGO_URI)
DB_NAME = os.getenv("MONGODB_DB_NAME")
db = client.get_default_database() if not DB_NAME else client[DB_NAME]
model_states = db["model_states"]

# Friendly name to HF repo mapping
MODEL_REPO_IDS: Dict[str, str] = {
    "7B-BASE": "username/7b-base",
    "67B-CHAT": "username/67b-chat",
    "LLAMA-3-8B": "meta-llama/Llama-3-8b",
    "MISTRAL-7B": "mistralai/mistral-7b",
    "FALCON-40B": "tiiuae/falcon-40b",
    "GPT-J-6B": "EleutherAI/gpt-j-6B",
    "DEEPSEEK-CODER-33B": "deepseek/coder-33b",
}

# ─────────────────────────────────────────────────────────────────────────────
# 2) FastAPI setup
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Ailo Forge", version="1.0.0", debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # return JSON with detail list
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# In-memory stores
progress_store: Dict[str, Dict[str, Any]] = {}
model_pipelines: Dict[str, Any]        = {}

# ─────────────────────────────────────────────────────────────────────────────
# 3) Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────
class ModifyRequest(BaseModel):
    model_version: str = Field(..., alias="modelVersion")
    temperature: float
    token_limit: int   = Field(..., alias="tokenLimit")
    instructions: str  = Field("", alias="instructions")

    class Config:
        allow_population_by_alias = True
        populate_by_name = True

class RunRequest(BaseModel):
    model_version: str = Field(..., alias="modelVersion")
    prompt: str

    class Config:
        allow_population_by_alias = True
        populate_by_name = True

# ─────────────────────────────────────────────────────────────────────────────
# 4) Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def healthcheck():
    return {"status": "ok", "message": "Ailo Forge backend live"}

@app.get("/models")
async def list_models():
    local = [d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d))]
    combined = list(dict.fromkeys(list(MODEL_REPO_IDS.keys()) + local))
    return {"models": combined}

@app.post("/modify-file")
async def modify_file(req: ModifyRequest):
    mv = req.model_version
    cfg_path = os.path.join("models", mv, "config.json")
    if not os.path.isfile(cfg_path):
        raise HTTPException(404, detail="Model folder or config.json missing")

    cfg = json.load(open(cfg_path))
    cfg["temperature"] = req.temperature
    cfg["tokenLimit"]    = req.token_limit
    cfg["instructions"]  = req.instructions
    json.dump(cfg, open(cfg_path, "w"), indent=2)

    model_pipelines.pop(mv, None)
    return {"success": True, "message": "Config updated"}

@app.post("/run")
async def run_model(req: RunRequest):
    mv = req.model_version
    prompt = req.prompt

    # determine source
    local_dir = os.path.join("models", mv)
    if os.path.isdir(local_dir):
        src = local_dir
    elif mv in MODEL_REPO_IDS and HF_HUB_TOKEN:
        src = snapshot_download(MODEL_REPO_IDS[mv], cache_dir="models")
    elif OPENAI_API_KEY:
        src = None
    else:
        raise HTTPException(404, detail="Model not found")

    if src:
        cfg     = AutoConfig.from_pretrained(src)
        temp    = getattr(cfg, "temperature", 0.7)
        max_len = getattr(cfg, "tokenLimit", 150)
        if mv not in model_pipelines:
            tok = AutoTokenizer.from_pretrained(src)
            mod = AutoModelForCausalLM.from_pretrained(src)
            model_pipelines[mv] = pipeline(
                "text-generation", model=mod, tokenizer=tok, device_map="auto"
            )
        out = model_pipelines[mv](prompt, max_length=max_len, temperature=temp)
        text = out[0]["generated_text"]
    else:
        resp = openai.chat.completions.create(
            model="gpt-4", messages=[{"role":"user","content":prompt}],
            temperature=0.7, max_tokens=150,
        )
        text = resp.choices[0].message.content.strip()

    return {"success": True, "response": text}

@app.post("/train")
async def train_model(
    base_model: str = Form(...), trainingObjective: str = Form(...),
    files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None,
):
    texts: List[str] = []
    for f in files:
        data = await f.read()
        if not data.startswith((b"\xFF\xD8", b"\x89PNG")):
            texts.append(data.decode(errors="ignore"))
    if not texts:
        raise HTTPException(400, detail="No valid text data provided")

    job_id = str(uuid.uuid4())
    progress_store[job_id] = {"percent":0, "status":"in_progress"}
    background_tasks.add_task(_run_training, job_id, base_model, texts)
    return {"job_id": job_id, "status": "training_started"}


def _run_training(job_id: str, base_model: str, texts: List[str]):
    try:
        if os.path.isdir(os.path.join("models", base_model)):
            model_id = os.path.join("models", base_model)
        elif base_model in MODEL_REPO_IDS and HF_HUB_TOKEN:
            model_id = snapshot_download(MODEL_REPO_IDS[base_model], cache_dir="models")
        else:
            raise RuntimeError("Base model not found and no HF_HUB_TOKEN")

        tok = AutoTokenizer.from_pretrained(model_id)
        mod = AutoModelForCausalLM.from_pretrained(model_id)
        ds  = Dataset.from_dict({"text":texts})
        ds  = ds.map(lambda x: tok(x["text"], truncation=True, max_length=128), batched=True)

        out_dir = os.path.join("models", f"{base_model}-ft-{job_id}")
        args = TrainingArguments(
            output_dir=out_dir, num_train_epochs=3, per_device_train_batch_size=2,
            logging_steps=10, save_steps=50,
            push_to_hub=bool(HF_HUB_TOKEN), hub_token=HF_HUB_TOKEN,
            hub_model_id=HF_HUB_REPO_ID or f"{base_model}-ft-{job_id}",
        )
        trainer = Trainer(model=mod, args=args, train_dataset=ds, tokenizer=tok)
        trainer.train(); trainer.save_model(out_dir)
        if HF_HUB_TOKEN: trainer.push_to_hub()
        progress_store[job_id] = {"percent":100, "status":"completed"}
    except Exception:
        logging.exception("Training failed")
        progress_store[job_id] = {"percent":0, "status":"failed"}

@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    if job_id in progress_store:
        return progress_store[job_id]
    raise HTTPException(404, detail="Job not found")

# ─────────────────────────────────────────────────────────────────────────────
# 5) Run Uvicorn
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT","8080")), reload=True)
