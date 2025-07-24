# NomNom-LLM

This is the LLM FastAPI backend. Chose fastapi because it allows to load the model before it accepts a request and is fast and lightweight. 

Notes and current issues:
Currently training different parameters to train my model on, dialogpt. Considering if I should choose another model instead but finetuning might be better for a faster/short context llm.

Current routes are limited, probably don't need too many considering general questions and ingredient substitutions are the only llm related use cases for now.

The app is pretty simple but solid:

Model loading: Uses an async lifespan context manager in FastAPI to load the base model + LoRA adapter once on startup. So no slow loading on every request.

Model: Currently based on DialoGPT-small for quick testing, but I’m training LoRA adapters on my own cooking Q&A data. Considering switching to stronger models later.

Text generation pipeline: Hugging Face pipeline handles tokenization, generation, and decoding in one place.

Routes:
- /generate: takes a cooking question + recipe + step and returns a helpful cooking answer.

- /ingredient-sub: takes an ingredient name and returns common substitutes.