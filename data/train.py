import torch
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# Configuration
BASE_MODEL = "microsoft/DialoGPT-small"
CSV_PATH = "cooking_knowledge.csv"
OUTPUT_DIR = "./cooking-lora-model"
BATCH_SIZE = 1
EPOCHS = 1 # run through data once
MAX_LENGTH = 256 # tokens in response

def main():
    print("Loading data...")
    # Load and format data
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["question", "response"])
    
    # keep input/output format for lora training
    dataset = Dataset.from_pandas(df[["question", "response"]])
    
    print("Loading model...")
    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16
    )
    
    lora_config = LoraConfig(
        r=12,  # level of change 8-16, factual information, lower change maybe 8-12
        lora_alpha=24,  # 2x the rank
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # tokenize dataset
    def tokenize(examples):
        # create input-output pairs
        inputs = [q + tokenizer.eos_token for q in examples["question"]]
        targets = [r + tokenizer.eos_token for r in examples["response"]]
        
        # tokenize inputs and targets
        model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=MAX_LENGTH)
        labels = tokenizer(targets, truncation=True, padding="max_length", max_length=MAX_LENGTH)
        
        # set labels (what we want the model to learn to output)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_dataset = dataset.map(tokenize, batched=True)
    
    # Split dataset (80% train, 20% eval)
    train_size = int(0.8 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
    
    print(f"Training samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    print("Starting training...")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        logging_steps=5,  # More frequent logging
        eval_strategy="no",  # Skip evaluation to save time
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=0  # Avoid multiprocessing issues on Mac
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    trainer.train()
    
    # Save
    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Testing trained model...")
    test_model(OUTPUT_DIR)
    
    print("Training complete!")

def test_model(model_path, test_question="What is food safety?"):
    """Test the trained model"""
    from peft import PeftModel
    
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # generate response
    inputs = tokenizer.encode(test_question + "<|endoftext|>", return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(test_question, "").strip()
    
    print(f"Test Question: {test_question}")
    print(f"Model Response: {response}")

if __name__ == "__main__":
    main()