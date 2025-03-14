"""
This script is used to evaluate a model on a dataset.
"""
import sys
import re
import json
import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser, AutoTokenizer, GenerationConfig
from datasets import disable_caching
from data.dataset import load_pt_dataset
import torch
from tqdm import tqdm
from models.llama_prompt_tuning import LlamaPromptTuningLM
from data.dataset import DatasetEnum, default_system_prompts, generation_lengths
from arguments import ModelArguments, DatasetArguments

disable_caching()

generation_config = GenerationConfig(
    do_sample=False,
    num_return_sequences=1,
)

@dataclass
class EvalArguments:
    """Arguments specific to evaluation."""
    output_file: str = field(
        metadata={"help": "Path to save evaluation results"}
    )
    soft_prompt_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the soft prompt checkpoint"}
    )
    eval_seed: Optional[int] = field(
        default=42,
        metadata={"help": "Seed for shuffling the evaluation dataset"}
    )
    batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "Batch size for evaluation"}
    )
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={"help": "Maximum sequence length for evaluation"}
    )

def format_prompt(input_text, system_prompt, tokenizer):
    """Format the prompt for the model."""

    messages = [
        {
            "role": "system",
            "content": ""
        },
        {
            "role": "user",
            "content": input_text
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    # Use regex to replace system prompt. <|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant for travel tips and recommendations<|eot_id|>
    text = re.sub(
        r"<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>[\s\S]*?<\|eot_id\|>", 
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>", 
        text
    )
    
    return text

def main():
    parser = HfArgumentParser((ModelArguments, DatasetArguments, EvalArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, dataset_args, eval_args = parser.parse_yaml_file(sys.argv[1])
    else:
        model_args, dataset_args, eval_args = parser.parse_args_into_dataclasses()
    dataset_name: DatasetEnum = dataset_args.dataset
    
    # Get system prompt
    system_prompt = model_args.system_prompt or default_system_prompts[dataset_name]
    results = []
    

    # Configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Initialize model
    model: LlamaPromptTuningLM = LlamaPromptTuningLM.from_pretrained(
        model_args.model_name,
        device_map="auto"
    )
    
    # Load soft prompt if using prompt tuning
    if model_args.use_prompt_tuning:
        model.load_soft_prompt(eval_args.soft_prompt_path)
    
    # model.init_soft_prompt_with_prompt_embedding(
    #     token_ids=tokenizer(system_prompt, add_special_tokens=False, return_tensors="pt").input_ids[0].to(model.device)
    # )
    
    print(f"Soft prompt length: {model.get_soft_prompt_len()}")
    
    model.eval()

    # Load dataset
    dataset = load_pt_dataset(dataset_name)
    test_dataset = dataset["test"]

    if dataset_args.eval_size and dataset_args.eval_size < len(test_dataset):
        # Shuffle the dataset
        test_dataset = test_dataset.shuffle(seed=eval_args.eval_seed)
        test_dataset = test_dataset.select(range(dataset_args.eval_size))
        
    # Get the length of the soft prompt
    soft_prompt_len = model.get_soft_prompt_len()

    # Evaluate
    batch_size = eval_args.batch_size
    generation_config.max_new_tokens = generation_lengths[dataset_name]
    for i in tqdm(range(0, len(test_dataset), batch_size)):
        examples = test_dataset[i:i+batch_size]
        texts = []
        for input_text in examples["input_text"]:
            if soft_prompt_len > 0:
                # Pad the soft prompt with padding tokens
                text = format_prompt(input_text, "<|end_of_text|>"*soft_prompt_len, tokenizer)
            else:
                text = format_prompt(input_text, system_prompt, tokenizer)
            texts.append(text)
        
        # Format input using chat template
        model_input = tokenizer(
            texts, 
            return_tensors="pt", 
            add_special_tokens=False,
            max_length=eval_args.max_seq_length,
            truncation=True,
            padding="longest"
        ).to(model.device)
        
        # Get the left padding offset using attention mask
        padding_offsets = (model_input.attention_mask == 0).sum(dim=1).cpu().tolist()
        

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **model_input,
                generation_config=generation_config,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # Use the padding offsets to generate the output
                padding_offsets=padding_offsets,
            )

        # Decode output
        generated_texts = tokenizer.batch_decode(
            outputs[:, len(model_input[0]):],
            skip_special_tokens=True
        )
        
        generated_texts = [text.strip() for text in generated_texts]

        for batch_idx in range(len(generated_texts)):
            # Store results
            result = {
                "dataset": dataset_name,
                "input_text": examples["input_text"][batch_idx],
                "model_output": generated_texts[batch_idx],
                "reference_answer": examples["answer"][batch_idx]
            }
            # If dataset is MBPP, remove the python code block
            if dataset_name == DatasetEnum.MBPP.value:
                result["model_output"] = result["model_output"].replace("```python\n", "").replace("```", "")
                # Also add all other columns to the result
            for key in examples.keys():
                result[key] = examples[key][batch_idx]
            results.append(result)

    # Save results
    # Create directory if it doesn't exist
    if not os.path.exists(os.path.dirname(eval_args.output_file)):
        os.makedirs(os.path.dirname(eval_args.output_file))
    with open(eval_args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
