from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.core.model_loader import lifespan
from api.routes import llm_qa

app = FastAPI(
    title="Cooking LLM API",
    description="A FastAPI backend with LoRA fine-tuned cooking LLM",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(llm_qa.router)
