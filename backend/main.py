#!/usr/bin/env python3
"""
Ailo Forge Backend (main.py)

Features:
  • Modify in-memory chat settings (temperature, tokens, instructions)
  • Chat via any Hugging Face model's Inference API (using HF_API_TOKEN) or fallback to OpenAI GPT-4
  • Fine-tune Hugging Face Hub models via Trainer with optional push to Hub
  • Pollable fine-tune progress endpoint
"""
import os
import logging
import uuid
from typing import List, Dict, Any

import requests
import openai
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load env & init
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO)

# OpenAI GPT-4 setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required for chat fallback")
openai.api_key = OPENAI_API_KEY
logging.info("Configured OpenAI GPT-4 client")

# Hugging Face API token for Inference & Hub
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    logging.warning("HF_API_TOKEN not set; HF Inference and push disabled")

# In-memory stores
chat_config: Dict[str, Any] = {}
train_progress: Dict[str, Dict[str, Any]] = {}

# FastAPI setup
app = FastAPI(title="Ailo Forge", version="4.0.0", debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
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
    model_id: str = Field(..., alias="modelId")
    prompt: str
    class Config:
        allow_population_by_alias = True
        allow_population_by_field_name = True

# ─────────────────────────────────────────────────────────────────────────────
# 3) Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def healthcheck():
    return {"status": "ok", "message": "Ailo Forge backend live"}

@app.post("/modify-chat")
async def modify_chat(req: ModifyChat):
    chat_config['settings'] = {
        "temperature": req.temperature,
        "max_tokens": req.token_limit,
        "instructions": req.instructions,
    }
    return {"success": True, "message": "Chat settings updated"}

@app.post("/run")
async def run_chat(req: RunChat):
    settings = chat_config.get('settings', {})
    temperature = settings.get('temperature', 0.7)
    max_tokens = settings.get('max_tokens', 150)
    instructions = settings.get('instructions', "")

    # Build OpenAI-style messages
    messages = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    messages.append({"role": "user", "content": req.prompt})

    # 1) Try Hugging Face Inference API
    if HF_API_TOKEN:
        hf_url = f"https://api-inference.huggingface.co/models/{req.model_id}"
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        payload = {
            "inputs": req.prompt,
            "parameters": {"temperature": temperature, "max_new_tokens": max_tokens},
        }
        try:
            hf_resp = requests.post(hf_url, headers=headers, json=payload, timeout=30)
            hf_resp.raise_for_status()
            data = hf_resp.json()
            # HF returns list or dict
            if isinstance(data, list) and 'generated_text' in data[0]:
                text = data[0]['generated_text']
            elif isinstance(data, dict) and 'generated_text' in data:
                text = data['generated_text']
            else:
                text = data.get('generated_text') or json.dumps(data)
            return {"success": True, "response": text}
        except Exception as e:
            logging.warning(f"HF Inference failed: {e}; falling back to OpenAI")

    # 2) Fallback to OpenAI GPT-4
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content.strip()
        return {"success": True, "response": text}
    except Exception as e:
        logging.error(f"OpenAI error: {e}")
        raise HTTPException(502, detail=str(e))

@app.post("/train")
async def train_model(
    repo_id: str = Form(...),
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    texts: List[str] = []
    for f in files:
        data = await f.read()
        if data.startswith((b"\xFF\xD8", b"\x89PNG")): continue
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
# 4) Background Training Function
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
            push_to_hub=bool(HF_API_TOKEN),
            hub_token=HF_API_TOKEN,
            hub_model_id=os.getenv("HF_HUB_REPO_ID") or f"ft-{job_id}",
        )
        trainer = Trainer(model=mod, args=args, train_dataset=ds, tokenizer=tok)
        trainer.train()
        trainer.save_model(out_dir)
        if HF_API_TOKEN:
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
