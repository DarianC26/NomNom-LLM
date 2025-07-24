from fastapi import APIRouter, HTTPException
from api.models.schemas import CookingQuery, CookingResponse, IngredientSubQuery, IngredientSubResponse
from api.core.generation_service import generate_cooking_response, generate_ingredient_substitutes
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
async def root():
    return {"message": "Cooking LLM API is running!", "status": "healthy"}

@router.get("/health")
async def health_check():
    from api.core.model_loader import text_generator  # local import to avoid circular
    model_status = "loaded" if text_generator else "not_loaded"
    return {"status": "healthy", "model_status": model_status, "version": "1.0.0"}

@router.post("/generate", response_model=CookingResponse)
async def generate_cooking_advice(query: CookingQuery):
    try:
        response = generate_cooking_response(
            query.recipe,
            query.step,
            query.prompt
        )
        return CookingResponse(response=response)
    except Exception as e:
        logger.error(f"Error in /generate: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating response")

@router.post("/ingredient-sub", response_model=IngredientSubResponse)
async def get_ingredient_substitutes(query: IngredientSubQuery):
    try:
        substitutes = generate_ingredient_substitutes(query.ingredient)
        return IngredientSubResponse(substitutes=substitutes)
    except Exception as e:
        logger.error(f"Error in /ingredient-sub: {str(e)}")
        raise HTTPException(status_code=500, detail="Error finding ingredient substitutes")
