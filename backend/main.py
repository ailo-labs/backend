#!/usr/bin/env python3
"""
Ailo Forge Backend (main.py)

Features:
  • List available models
  • Modify in-memory model generation settings
  • Chat with HF Hub models via remote inference API, or fallback to OpenAI GPT-4
  • Fine-tune HF Hub models via Hugging Face Trainer with checkpointing & optional Hub push
  • Pollable progress endpoint
"""
import os
import logging
import uuid
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# OpenAI v1 client
from openai import OpenAI
# HF Inference API
from huggingface_hub import InferenceApi, login as hf_login
# Transformers for training
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load env & init
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO)

# HF Hub auth and inference client
HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN")
if HF_HUB_TOKEN:
    hf_login(HF_HUB_TOKEN)
    logging.info("Logged into HF Hub")
    # create inference clients for each model
else:
    logging.warning("HF_HUB_TOKEN not set; HF inference disabled")

# OpenAI client
client_openai: Optional[OpenAI] = None
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    client_openai = OpenAI(api_key=OPENAI_KEY)
    logging.info("OpenAI client configured")
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

# instantiate inference clients map
inference_clients: Dict[str, InferenceApi] = {}
if HF_HUB_TOKEN:
    for key, repo in MODEL_REPO_IDS.items():
        inference_clients[key] = InferenceApi(repo, token=HF_HUB_TOKEN)

# In-memory stores
config_store: Dict[str, Dict[str, Any]] = {}
progress_store: Dict[str, Dict[str, Any]] = {}

# Create app
app = FastAPI(title="Ailo Forge", version="1.0.0", debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
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
        allow_population_by_field_name = True

class RunRequest(BaseModel):
    model_version: str = Field(..., alias="modelVersion")
    prompt: str
    class Config:
        allow_population_by_alias = True
        allow_population_by_field_name = True

# Endpoints
@app.get("/")
async def healthcheck():
    return {"status": "ok", "message": "Ailo Forge live"}

@app.get("/models")
async def list_models():
    return {"models": list(MODEL_REPO_IDS.keys())}

@app.post("/modify-file")
async def modify_file(req: ModifyRequest):
    mv = req.model_version
    if mv not in MODEL_REPO_IDS:
        raise HTTPException(404, detail="Unknown model version")
    config_store[mv] = {
        "temperature": req.temperature,
        "token_limit": req.token_limit,
        "instructions": req.instructions,
    }
    return {"success": True, "message": "Config updated"}

@app.post("/run")
async def run_model(req: RunRequest):
    mv, prompt = req.model_version, req.prompt
    cfg = config_store.get(mv, {})
    temp = cfg.get("temperature", 0.7)
    max_len = cfg.get("token_limit", 150)
    instr = cfg.get("instructions", "")

    # HF Inference API path
    client = inference_clients.get(mv)
    if client:
        full_prompt = instr + "\n" + prompt if instr else prompt
        try:
            resp = client(inputs=full_prompt, parameters={"max_new_tokens": max_len, "temperature": temp})
            text = resp.get("generated_text") or resp.get("choices", [{}])[0].get("text", "")
        except Exception as e:
            logging.error(f"HF Inference failed for {mv}: {e}")
            raise HTTPException(502, detail=f"HF inference failed: {e}")
    # OpenAI fallback
    elif client_openai:
        resp = client_openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role":"system","content":instr}, {"role":"user","content":prompt}],
            temperature=temp,
            max_tokens=max_len,
        )
        text = resp.choices[0].message.content.strip()
    else:
        raise HTTPException(404, detail="No valid inference source available")

    return {"success": True, "response": text}

@app.post("/train")
async def train_model(
    base_model: str = Form(...), objective: str = Form(...),
    files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None
):
    if base_model not in MODEL_REPO_IDS:
        raise HTTPException(404, detail="Unknown model version")
    texts = []
    for f in files:
        data = await f.read()
        if data.startswith((b"\xFF\xD8", b"\x89PNG")): continue
        texts.append(data.decode("utf-8", errors="ignore"))
    if not texts:
        raise HTTPException(400, detail="No valid text data provided")
    job_id = str(uuid.uuid4())
    progress_store[job_id] = {"percent": 0, "status": "in_progress"}
    background_tasks.add_task(_run_training, job_id, base_model, texts, objective)
    return {"job_id": job_id, "status": "in_progress"}

# Background training

def _run_training(job_id: str, base_model: str, texts: List[str], objective: str):
    try:
        repo = MODEL_REPO_IDS[base_model]
        tok = AutoTokenizer.from_pretrained(repo)
        mod = AutoModelForCausalLM.from_pretrained(repo)
        ds = Dataset.from_dict({"text":texts})
        ds = ds.map(lambda x: tok(x["text"], truncation=True, max_length=128), batched=True)
        out_dir = os.path.join("models", f"{base_model}-ft-{job_id}")
        args = TrainingArguments(
            output_dir=out_dir, num_train_epochs=3, per_device_train_batch_size=2,
            logging_steps=10, save_steps=50,
            push_to_hub=bool(HF_HUB_TOKEN), hub_token=HF_HUB_TOKEN,
            hub_model_id=os.getenv("HF_HUB_REPO_ID") or f"{base_model}-ft-{job_id}"
        )
        trainer = Trainer(model=mod, args=args, train_dataset=ds, tokenizer=tok)
        trainer.train()
        trainer.save_model(out_dir)
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT","8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
