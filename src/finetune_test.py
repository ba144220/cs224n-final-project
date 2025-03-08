from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
import torch
import numpy as np
from models.llama_prompt_tuning import LlamaPromptTuningConfig, LlamaPromptTuningLM

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

config = LlamaPromptTuningConfig.from_pretrained(MODEL_NAME)
config.prompt_tuning_range = (5, 48)

# Configure the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


model: LlamaPromptTuningLM = LlamaPromptTuningLM.from_pretrained(MODEL_NAME, config=config, device_map="auto")
# model.init_soft_prompt_with_prompt_embedding(
#     token_ids=tokenizer(
#         "Cutting Knowledge Date: December 2023\nToday Date: 25 Feb 2025\n\nYou are a helpful assistant that answers questions based on the provided context. You can only answer with Yes or No.",
#         add_special_tokens=False,
#         return_tensors="pt"
#     ).input_ids[0].to(model.device)
# )

model.init_soft_prompt_with_random_values()

dataset = load_dataset("super_glue", "boolq")

def format_boolq_to_dialog(example):
    """Convert BoolQ examples to Llama chat format"""
    context = example['passage']
    question = example['question']
    answer = "Yes" if example['label'] else "No"
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based on the provided context. You can only answer with Yes or No."
        },
        {
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {question}"
        },
        {
            "role": "assistant",
            "content": answer
        }
    ]
    
    return {"messages": messages}

# Process the dataset
train_dataset = dataset['train'].map(format_boolq_to_dialog).shuffle(seed=42).select(range(1000))
eval_dataset = dataset['validation'].map(format_boolq_to_dialog).select(range(100))

def tokenize_function(examples):
    texts = [
        tokenizer.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=False
        ) for msg in examples["messages"]
    ]
    
    return tokenizer(
        texts,
        return_tensors="pt",
        add_special_tokens=False,
        padding="longest"
    )

# Tokenize datasets
train_dataset = train_dataset.map(
    tokenize_function,
    remove_columns=train_dataset.column_names,
    batched=True,
    batch_size=1
)
eval_dataset = eval_dataset.map(
    tokenize_function,
    remove_columns=eval_dataset.column_names,
    batched=True,
    batch_size=1
)

# Set up data collator for completion-only learning
response_template = [128006, 78191, 128007, 271]
# response_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./output/random_init",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    # evaluation_strategy="steps",
    # eval_steps=200,
    save_strategy="steps",
    save_steps=400,
    # load_best_model_at_end=True,
    save_total_limit=1,
    learning_rate=2e-5,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
)

# Start training
trainer.train()

# # Save the trained model
# model.save_pretrained("./boolq_prompt_tuned_model")
