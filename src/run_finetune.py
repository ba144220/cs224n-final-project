"""
This script is used to train a model on a dataset using prompt tuning.
"""
import sys
import os
import re
from transformers import (
    HfArgumentParser, 
    AutoTokenizer,
    TrainerCallback
)

from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig
from datasets import disable_caching

from data.dataset import (
    load_pt_dataset,
    default_system_prompts
)
from models.llama_prompt_tuning import LlamaPromptTuningLM
from arguments import ModelArguments, DatasetArguments, TrainingArguments

disable_caching()

def format_prompt(example, system_prompt, tokenizer):
    """Format the prompt for the model."""
    user_prompt = example["input_text"]

    messages = [
        {
            "role": "system",
            "content": ""
        },
        {
            "role": "user",
            "content": user_prompt
        },
        {
            "role": "assistant",
            "content": example["answer"]
        }
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    text = re.sub(
        r"<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>[\s\S]*?<\|eot_id\|>", 
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>", 
        text
    )
    return text

def tokenize_function(examples, tokenizer):
    """Tokenize the examples."""
    return tokenizer(
        examples["text"],
        return_tensors="pt",
        add_special_tokens=False,
        padding="longest",
        truncation=True,
        max_length=2048
    )

class SoftPromptSaveCallback(TrainerCallback):
    """Callback to save soft prompt during training."""
    def on_save(self, args, state, control, model, **kwargs):
        if args.should_save:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_soft_prompt(checkpoint_dir)
            control.should_save = False

def main():
    parser = HfArgumentParser((ModelArguments, DatasetArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, dataset_args, training_args = parser.parse_yaml_file(sys.argv[1])
    else:
        model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()

    dataset_name = dataset_args.dataset
    
    # Get system prompt
    system_prompt = model_args.system_prompt or default_system_prompts[dataset_name]

    # Configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Initialize model
    model: LlamaPromptTuningLM = LlamaPromptTuningLM.from_pretrained(
        model_args.model_name,
        device_map="auto"
    )

    # Initialize soft prompt based on configuration
    if model_args.init_from_pretrained:
        model.load_soft_prompt(model_args.init_from_pretrained)
    elif model_args.init_from_natural_language:
        model.init_soft_prompt_with_prompt_embedding(
            token_ids=tokenizer(system_prompt, add_special_tokens=False, return_tensors="pt").input_ids[0].to(model.device),
            soft_prompt_len=model_args.prompt_tuning_length
        )
    elif model_args.init_random:
        model.init_soft_prompt_with_random_values(
            soft_prompt_len=model_args.prompt_tuning_length
        )
    else:
        raise ValueError("Must specify one of init_from_pretrained, init_from_natural_language, or init_random")
    
    print(f"Soft prompt length: {model.get_soft_prompt_len()}")
    
    soft_prompt_len = model.get_soft_prompt_len()

    # Load dataset
    dataset = load_pt_dataset(dataset_name)
    
    # Apply size limits if specified
    if dataset_args.train_size:
        # Shuffle the dataset
        dataset["train"] = dataset["train"].shuffle(seed=42)
        dataset["train"] = dataset["train"].select(range(min(dataset_args.train_size, len(dataset["train"]))))
    if dataset_args.eval_size:
        dataset["validation"] = dataset["validation"].shuffle(seed=42)
        dataset["validation"] = dataset["validation"].select(range(min(dataset_args.eval_size, len(dataset["validation"]))))

    # Format datasets
    def format_dataset(example):
        if soft_prompt_len > 0:
            formatted_texts = format_prompt(example, "<|end_of_text|>"*soft_prompt_len, tokenizer)
        else:
            formatted_texts = format_prompt(example, system_prompt, tokenizer)
        example["text"] = formatted_texts
        return example
    train_remove_columns = dataset["train"].column_names
    if "text" in train_remove_columns:
        train_remove_columns = train_remove_columns.remove("text")
    eval_remove_columns = dataset["validation"].column_names
    if "text" in eval_remove_columns:
        eval_remove_columns = eval_remove_columns.remove("text")
        
    train_dataset = dataset["train"].map(
        format_dataset,
        batched=False,
        remove_columns=train_remove_columns
    )
    eval_dataset = dataset["validation"].map(
        format_dataset,
        batched=False,
        remove_columns=eval_remove_columns
    )
    
    print(train_dataset)
    print(eval_dataset)

    # Set up data collator
    response_template = [128006, 78191, 128007, 271]
    
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    
    sft_config = SFTConfig(
        output_dir=training_args.output_dir,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        num_train_epochs=training_args.num_train_epochs,
        max_steps=training_args.max_steps,
        
        logging_steps=training_args.logging_steps,
        max_seq_length=training_args.max_seq_length,
        
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        warmup_steps=training_args.warmup_steps,
        
        do_train=training_args.do_train,
        do_eval=training_args.do_eval,
        eval_strategy=training_args.eval_strategy,
        eval_steps=training_args.eval_steps,
        eval_on_start=training_args.eval_on_start,
        
        save_strategy=training_args.save_strategy,
        save_total_limit=training_args.save_total_limit,
        save_steps=training_args.save_steps,
        
        label_names=["labels"],
    )
    
    # Initialize trainer with callback
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[SoftPromptSaveCallback()],
    )

    # Train
    trainer.train()

    # Save final soft prompt
    model.save_soft_prompt(training_args.output_dir)

if __name__ == "__main__":
    main()
