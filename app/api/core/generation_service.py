from api.core.model_loader import text_generator, tokenizer
import logging

logger = logging.getLogger(__name__)

def generate_cooking_response(recipe: str, step: str, prompt: str, **kwargs) -> str:
    cooking_prompt = (
        f"As a cooking expert, a home cook is following this recipe: {recipe} "
        f"and they are on this step: {step}. The home cook has this general question: {prompt}. "
        f"Respond in a way to help resolve their problem:"
    )

    try:
        outputs = text_generator(
            cooking_prompt,
            max_length=kwargs.get('max_length', 200),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.9),
            do_sample=kwargs.get('do_sample', True),
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
        return outputs[0]["generated_text"].replace(cooking_prompt, "").strip()
    except Exception as e:
        logger.error(f"Error generating cooking response: {str(e)}")
        return "Sorry, I couldn't generate a helpful response."

def generate_ingredient_substitutes(ingredient: str) -> list[str]:
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

    if not text_generator:
        return ["No common substitutes found"]

    substitute_prompt = f"List ingredient substitutes for {ingredient}. Respond with only comma separated ingredients:"
    try:
        outputs = text_generator(
            substitute_prompt,
            max_length=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
        raw_text = outputs[0]['generated_text'].replace(substitute_prompt, "").strip()
        return [s.strip() for s in raw_text.split(',')][:5]
    except Exception as e:
        logger.error(f"Error generating ingredient substitutes: {str(e)}")
        return [f"Unable to find substitutes for {ingredient}"]
