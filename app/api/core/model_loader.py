import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
from contextlib import asynccontextmanager
import os
import logging

logger = logging.getLogger(__name__)

model = None
tokenizer = None
text_generator = None

@asynccontextmanager
async def lifespan(app):
    global model, tokenizer, text_generator

    try:
        logger.info("Loading cooking LLM model...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        base_model_name = "microsoft/DialoGPT-small"
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        lora_model_path = "./cooking-lora-model"
        if os.path.exists(lora_model_path):
            logger.info(f"Loading LoRA adapter from {lora_model_path}")
            model = PeftModel.from_pretrained(base_model, lora_model_path)
        else:
            logger.warning("LoRA model not found. Using base model.")
            model = base_model

        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        text_generator = None

    yield
