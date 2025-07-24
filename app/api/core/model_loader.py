import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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

        base_model_name = "microsoft/DialoGPT-small"
        print(f"Loading tokenizer from {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded successfully")

        print(f"Loading base model from {base_model_name}")  
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        print("Base model loaded successfully")

        # Get absolute path and check
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lora_model_path = os.path.join(current_dir, "cooking-lora-model")
        print(f"Looking for LoRA model at: {lora_model_path}")
        print(f"LoRA model exists: {os.path.exists(lora_model_path)}")
        
        if os.path.exists(lora_model_path):
            logger.info(f"Loading LoRA adapter from {lora_model_path}")
            print("Loading LoRA adapter...")
            model = PeftModel.from_pretrained(base_model, lora_model_path)
            print("LoRA adapter loaded successfully")
        else:
            logger.warning("LoRA model not found. Using base model.")
            model = base_model

        print("Creating text generation pipeline...")
        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16
        )
        print("Pipeline created successfully")
        print(text_generator)

        logger.info("Model loaded successfully.")
    except Exception as e:
        print(f"Detailed error: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        logger.error(f"Error loading model: {str(e)}")
        text_generator = None

    yield

def get_text_generator():
    """Function to get the initialized text generator"""
    return text_generator

def get_tokenizer():
    """Function to get the initialized tokenizer"""  
    return tokenizer