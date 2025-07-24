from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel
import uvicorn
import logging
import os
from contextlib import asynccontextmanager
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
text_generator = None

# Pydantic models
class CookingQuery(BaseModel):
    prompt: str
    recipe: str
    step: str

class CookingResponse(BaseModel):
    response: str

class IngredientSubQuery(BaseModel):
    ingredient: str

class IngredientSubResponse(BaseModel):
    substitutes: List[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    global model, tokenizer, text_generator
    
    try:
        logger.info("Loading cooking LLM model...")
        
        # Configuration for efficient loading
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Base model - using a smaller model for demo purposes
        # In production, you'd use your specific cooking model
        base_model_name = "microsoft/DialoGPT-small"  # Replace with your base model
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA adapter (replace with your cooking LoRA model path)
        lora_model_path = "./cooking-lora-model"  # Update this path
        
        # Check if LoRA model exists, if not use base model
        if os.path.exists(lora_model_path):
            logger.info(f"Loading LoRA adapter from {lora_model_path}")
            model = PeftModel.from_pretrained(base_model, lora_model_path)
        else:
            logger.warning(f"LoRA model not found at {lora_model_path}, using base model")
            model = base_model
        
        # Create text generation pipeline
        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Fallback: create a mock generator for development
        text_generator = None
        
    yield
    
    # Shutdown
    logger.info("Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="Cooking LLM API",
    description="A FastAPI backend with LoRA fine-tuned cooking LLM",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to generate cooking responses
def generate_cooking_response(recipe: str, step: str, prompt: str, **kwargs) -> str:
    """Generate cooking-related response using the loaded model"""

    # This is a cooking prompt for general questions
    cooking_prompt = f"As a cooking expert, a home cook is following this recipe: {recipe} \
        and they are on this step: {step}. The home cook has this general question: {prompt}. \
            Respond in a way to help resolve their problem:"
    
    try:
        # Generate response
        outputs = text_generator(
            cooking_prompt,
            max_length=kwargs.get('max_length', 200),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.9),
            do_sample=kwargs.get('do_sample', True),
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
        
        # Extract generated text
        generated_text = outputs[0]['generated_text']
        response = generated_text.replace(cooking_prompt, "").strip()
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I apologize, but I'm having trouble generating a response right now. Please try again."

def generate_ingredient_substitutes(ingredient: str) -> List[str]:
    """Generate ingredient substitutes using the loaded model"""
    
    # prompt specifically for ingredient substitution
    substitute_prompt = f"List ingredient substitutes for {ingredient}. \
        Respond with only comma seperated ingredients: "
    
    try:
        if text_generator is None:
            # basic mvp, just have commonly substituted ingredients for demoing/speed
            fallback_subs = {
                "butter": ["margarine", "coconut oil", "vegetable oil", "applesauce"],
                "eggs": ["flax eggs", "chia eggs", "applesauce", "mashed banana"],
                "milk": ["almond milk", "soy milk", "oat milk", "coconut milk"],
                "flour": ["almond flour", "coconut flour", "rice flour", "oat flour"],
                "sugar": ["honey", "maple syrup", "stevia", "coconut sugar"],
                "salt": ["sea salt", "garlic powder", "onion powder", "herbs"],
                "vanilla": ["almond extract", "lemon extract", "rum extract", "vanilla paste"]
            }
            
            ingredient_lower = ingredient.lower()
            for key in fallback_subs:
                if key in ingredient_lower:
                    return fallback_subs[key]
            
            return ["No common substitutes found"]
        
        # generate response using the model
        outputs = text_generator(
            substitute_prompt,
            max_length=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
        
        # extract text
        generated_text = outputs[0]['generated_text']
        response = generated_text.replace(substitute_prompt, "").strip()
        
        # parse the ingredients by line
        substitutes = response.split(',')
        substitutes = []
        
        return substitutes[:5]  # limiting to 5
        
    except Exception as e:
        logger.error(f"Error generating ingredient substitutes: {str(e)}")
        return [f"Unable to find substitutes for {ingredient}"]

@app.get("/")
async def root():
    return {"message": "Cooking LLM API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    model_status = "loaded" if text_generator is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "version": "1.0.0"
    }

@app.post("/generate", response_model=CookingResponse)
async def generate_cooking_advice(query: CookingQuery):
    """Generate general cooking advice or answer cooking questions"""
    
    try:
        response = generate_cooking_response(
            query.prompt,
            max_length=query.max_length,
            temperature=query.temperature,
            top_p=query.top_p,
            do_sample=query.do_sample
        )
        
        return CookingResponse(response=response)
        
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating response")

@app.post("/ingredient-sub", response_model=IngredientSubResponse)
async def get_ingredient_substitutes(query: IngredientSubQuery):
    """Get substitutes for a specific ingredient"""
    
    try:
        substitutes = generate_ingredient_substitutes(query.ingredient)
        return IngredientSubResponse(substitutes=substitutes)
        
    except Exception as e:
        logger.error(f"Error in ingredient-sub endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Error finding ingredient substitutes")

if __name__ == "__main__":
    # For development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )