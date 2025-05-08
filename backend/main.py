#!/usr/bin/env python3
"""
Ailo Forge Backend (main.py)

Features:
  • Modify model generation settings in-place
  • Chat with local HF models, or fallback to OpenAI GPT-4
  • Fine-tune base models via Hugging Face Trainer with real checkpointing & optional Hub push
  • Pollable progress endpoint
"""

import os
import logging
import uuid
import json
import asyncio
from typing import List, Dict

from fastapi          import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic         import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv           import load_dotenv

import openai
from transformers     import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    Trainer,
    TrainingArguments,
)
from datasets         import Dataset
from huggingface_hub  import login as hf_login

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load env & initialize
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO)

# OpenAI GPT-4 fallback
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key  = OPENAI_API_KEY
if not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY not set; GPT-4 fallback disabled")

# Hugging Face Hub (for push_to_hub and private models)
HF_HUB_TOKEN    = os.getenv("HF_HUB_TOKEN")
HF_HUB_REPO_ID  = os.getenv("HF_HUB_REPO_ID")
if HF_HUB_TOKEN:
    hf_login(HF_HUB_TOKEN)

# MongoDB (for storing state if you need it)
MONGO_URI       = os.getenv("MONGO_URI") or ""
if not MONGO_URI:
    raise RuntimeError("MONGO_URI is required")
client          = AsyncIOMotorClient(MONGO_URI)
DB_NAME         = os.getenv("MONGODB_DB_NAME")
db              = client.get_default_database() if not DB_NAME else client[DB_NAME]
model_states    = db["model_states"]

# ─────────────────────────────────────────────────────────────────────────────
# 2) FastAPI setup
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Ailo Forge", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

progress_store: Dict[str, Dict[str, any]] = {}
model_pipelines: Dict[str, any]        = {}

class ModifyRequest(BaseModel):
    model_version: str
    temperature:   float
    tokenLimit:    int
    instructions:  str = ""

@app.get("/")
async def healthcheck():
    return {"status": "ok", "message": "Ailo Forge backend live"}

# ─────────────────────────────────────────────────────────────────────────────
# 3) Modify generation settings
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/modify-file")
async def modify_file(req: ModifyRequest):
    local_dir = os.path.join("models", req.model_version)
    cfg_path  = os.path.join(local_dir, "config.json")
    if not os.path.isdir(local_dir) or not os.path.isfile(cfg_path):
        raise HTTPException(404, detail="Local model folder or config.json missing")
    cfg = json.load(open(cfg_path))
    # Overwrite HF config fields (you can add others)
    cfg["temperature"]  = req.temperature
    cfg["tokenLimit"]   = req.tokenLimit
    cfg["instructions"] = req.instructions
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    # Clear cached pipeline so new settings reload
    model_pipelines.pop(req.model_version, None)
    return {"success": True, "message": "Generation config updated"}

# ─────────────────────────────────────────────────────────────────────────────
# 4) Chat endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/run")
async def run_model(payload: Dict):
    mv     = payload.get("model_version")
    prompt = payload.get("prompt")
    if not mv or not prompt:
        raise HTTPException(400, detail="model_version and prompt are required")

    local_dir = os.path.join("models", mv)
    if os.path.isdir(local_dir):
        # Load saved defaults
        cfg     = AutoConfig.from_pretrained(local_dir)
        temp    = getattr(cfg, "temperature", 0.7)
        max_len = getattr(cfg, "tokenLimit", 150)

        if mv not in model_pipelines:
            tok = AutoTokenizer.from_pretrained(local_dir)
            mod = AutoModelForCausalLM.from_pretrained(local_dir)
            model_pipelines[mv] = pipeline("text-generation", model=mod, tokenizer=tok, device_map="auto")
        gen = model_pipelines[mv]
        out = gen(prompt, max_length=max_len, temperature=temp)
        text = out[0]["generated_text"]

    elif OPENAI_API_KEY:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role":"user","content":prompt}],
            temperature=0.7,
            max_tokens=150,
        )
        text = resp.choices[0].message.content.strip()
    else:
        text = prompt[::-1]  # fallback

    return {"success": True, "response": text}

# ─────────────────────────────────────────────────────────────────────────────
# 5) Fine-tune endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/train")
async def train_model(
    base_model:        str            = Form(...),
    trainingObjective: str            = Form(...),
    files:             List[UploadFile] = File(...),
    background_tasks:  BackgroundTasks = None,
):
    # Collect text from uploads
    texts = []
    for f in files:
        data = await f.read()
        if not (data.startswith(b"\xFF\xD8") or data.startswith(b"\x89PNG")):
            texts.append(data.decode("utf-8", errors="ignore"))
    if not texts:
        raise HTTPException(400, detail="No valid text data provided")

    job_id = str(uuid.uuid4())
    progress_store[job_id] = {"percent": 0, "status": "in_progress"}
    background_tasks.add_task(_run_training, job_id, base_model, texts)
    return {"job_id": job_id, "status": "training_started"}

def _run_training(job_id: str, base_model: str, texts: List[str]):
    try:
        # Determine model source
        local_dir = os.path.join("models", base_model)
        if os.path.isdir(local_dir):
            model_id = local_dir
        elif HF_HUB_TOKEN:
            model_id = base_model  # HF Hub repo_id (e.g. "your-user/7B-Base")
        else:
            raise RuntimeError("Base model not found locally or no HF_HUB_TOKEN")

        tok   = AutoTokenizer.from_pretrained(model_id)
        mod   = AutoModelForCausalLM.from_pretrained(model_id)
        ds    = Dataset.from_dict({"text": texts})
        ds    = ds.map(lambda x: tok(x["text"], truncation=True, max_length=128), batched=True)

        out_dir = os.path.join("models", f"{base_model}-ft-{job_id}")
        args    = TrainingArguments(
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

# ─────────────────────────────────────────────────────────────────────────────
# 6) Progress endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    if job_id in progress_store:
        return progress_store[job_id]
    raise HTTPException(404, detail="Job not found")

# ─────────────────────────────────────────────────────────────────────────────
# 7) Run if invoked directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
