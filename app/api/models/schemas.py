from pydantic import BaseModel
from typing import List

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
