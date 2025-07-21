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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
text_generator = None

# Pydantic models for request/response
class CookingQuery(BaseModel):
    prompt: str
    max_length: Optional[int] = 200
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    do_sample: Optional[bool] = True

class RecipeRequest(BaseModel):
    ingredients: List[str]
    cuisine_type: Optional[str] = "any"
    dietary_restrictions: Optional[List[str]] = []
    difficulty: Optional[str] = "medium"

class TechniqueQuery(BaseModel):
    technique: str
    context: Optional[str] = None

class CookingResponse(BaseModel):
    response: str
    confidence: Optional[float] = None

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
def generate_cooking_response(prompt: str, **kwargs) -> str:
    """Generate cooking-related response using the loaded model"""
    
    if text_generator is None:
        # Mock responses for development/demo
        mock_responses = {
            "sauté": "Sautéing involves cooking food quickly in a small amount of fat over high heat while stirring frequently. The key is to keep ingredients moving to prevent burning while achieving a golden color and tender texture.",
            "braise": "Braising combines dry and wet heat cooking. First, sear the protein to develop flavor, then add liquid and cook slowly at low temperature. This technique transforms tough cuts into tender, flavorful dishes.",
            "recipe": "Here's a simple technique: Always preheat your pan before adding oil, then add ingredients when the oil shimmers. This prevents sticking and ensures even cooking.",
            "knife skills": "Proper knife technique starts with the grip - pinch the blade with thumb and forefinger, curl fingers of guide hand. Rock the knife, don't chop straight down."
        }
        
        for key, response in mock_responses.items():
            if key.lower() in prompt.lower():
                return response
        
        return "I'm a cooking assistant focused on techniques, recipes, and culinary advice. How can I help you improve your cooking skills today?"
    
    # Format prompt for cooking context
    cooking_prompt = f"As a cooking expert, {prompt}\n\nResponse:"
    
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

@app.post("/recipe", response_model=CookingResponse)
async def get_recipe_suggestions(recipe_req: RecipeRequest):
    """Get recipe suggestions based on ingredients and preferences"""
    
    # Build prompt
    ingredients_str = ", ".join(recipe_req.ingredients)
    restrictions_str = ", ".join(recipe_req.dietary_restrictions) if recipe_req.dietary_restrictions else "none"
    
    prompt = f"""
    Please suggest a {recipe_req.difficulty} difficulty {recipe_req.cuisine_type} recipe using these ingredients: {ingredients_str}.
    Dietary restrictions: {restrictions_str}.
    Include cooking techniques and brief instructions.
    """
    
    try:
        response = generate_cooking_response(prompt.strip())
        return CookingResponse(response=response)
        
    except Exception as e:
        logger.error(f"Error in recipe endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating recipe")

@app.post("/technique", response_model=CookingResponse)
async def explain_cooking_technique(technique_query: TechniqueQuery):
    """Explain a specific cooking technique"""
    
    context_part = f" in the context of {technique_query.context}" if technique_query.context else ""
    prompt = f"Explain the cooking technique '{technique_query.technique}'{context_part}. Include tips for success and common mistakes to avoid."
    
    try:
        response = generate_cooking_response(prompt)
        return CookingResponse(response=response)
        
    except Exception as e:
        logger.error(f"Error in technique endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Error explaining technique")

@app.get("/techniques")
async def list_cooking_techniques():
    """List common cooking techniques"""
    techniques = {
        "dry_heat": [
            "Sautéing", "Pan-frying", "Deep-frying", "Roasting", 
            "Baking", "Grilling", "Broiling", "Smoking"
        ],
        "moist_heat": [
            "Boiling", "Simmering", "Poaching", "Steaming", 
            "Braising", "Stewing", "Sous vide"
        ],
        "combination": [
            "Braising", "Stewing", "Fricassee"
        ],
        "knife_techniques": [
            "Julienne", "Brunoise", "Chiffonade", "Dice", "Mince"
        ]
    }
    
    return {"cooking_techniques": techniques}

@app.get("/tips/{category}")
async def get_cooking_tips(category: str):
    """Get cooking tips for specific categories"""
    
    prompt = f"Provide 5 essential cooking tips for {category.replace('_', ' ')} cooking. Be specific and practical."
    
    try:
        response = generate_cooking_response(prompt)
        return CookingResponse(response=response)
        
    except Exception as e:
        logger.error(f"Error getting tips: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting cooking tips")

if __name__ == "__main__":
    # For development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )