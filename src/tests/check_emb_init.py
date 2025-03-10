import sys
import os

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_dir)

from models.llama_prompt_tuning import LlamaPromptTuningConfig, LlamaPromptTuningLM
from transformers import LlamaForCausalLM
from typing import Tuple
import torch

def load_model(
    model_name: str,
    prompt_tuning_range: Tuple[int, int],
) -> Tuple[LlamaPromptTuningLM, LlamaPromptTuningConfig]:
    config = LlamaPromptTuningConfig.from_pretrained(model_name)
    config.prompt_tuning_range = prompt_tuning_range
    model = LlamaPromptTuningLM.from_pretrained(model_name, config=config)
    model.cuda()
    return model, config

def test_init_soft_prompt_with_zero(
    pt_model: LlamaPromptTuningLM,
):
    pt_model.init_soft_prompt_with_zero()
    pt_len = pt_model.config.prompt_tuning_range[1] - pt_model.config.prompt_tuning_range[0]
    # Check the shape of the soft prompt
    if pt_model.get_soft_prompt().shape != (pt_len, pt_model.config.hidden_size):
        print(f"Shape of the soft prompt is incorrect: {pt_model.get_soft_prompt().shape}")
        return False
    # Check the values of the soft prompt all close to 0
    soft_prompt = pt_model.get_soft_prompt()
    if not torch.allclose(soft_prompt.data, torch.zeros_like(soft_prompt.data)):
        print(f"Values of the soft prompt are not close to 0: {soft_prompt.data}")
        return False
    
    return True

def test_init_soft_prompt_with_random_values(
    pt_model: LlamaPromptTuningLM,
):
    # Generate random mean and std
    mean = torch.rand(1).item()
    # Generate random std between 0 and 1
    std = torch.rand(1).item()
    pt_model.init_soft_prompt_with_random_values(
        mean=mean,
        std=std,
    )
    # Check the shape of the soft prompt
    pt_len = pt_model.config.prompt_tuning_range[1] - pt_model.config.prompt_tuning_range[0]
    if pt_model.get_soft_prompt().shape != (pt_len, pt_model.config.hidden_size):
        print(f"Shape of the soft prompt is incorrect: {pt_model.get_soft_prompt().shape}")
        return False
        
    # Check the mean and std of the soft prompt
    if not abs(pt_model.get_soft_prompt().mean() - mean) < 0.01:
        print(f"Mean of the soft prompt is incorrect: {pt_model.get_soft_prompt().mean()}")
        return False
    if not abs(pt_model.get_soft_prompt().std() - std) < 0.01:
        print(f"Std of the soft prompt is incorrect: {pt_model.get_soft_prompt().std()}")
        return False
        
    return True

def test_init_soft_prompt_with_prompt_embedding(
    pt_model: LlamaPromptTuningLM,
    base_model: LlamaForCausalLM,
):
    pt_start = pt_model.config.prompt_tuning_range[0]
    pt_end = pt_model.config.prompt_tuning_range[1]
    pt_len = pt_end - pt_start
    
    # Generate random token ids
    pt_token_ids = torch.randint(0, base_model.config.vocab_size, (pt_len,))
    bg_token_ids = torch.randint(0, base_model.config.vocab_size, (1, pt_len*2)) # 1 is the batch size
    
    pt_model.init_soft_prompt_with_prompt_embedding(pt_token_ids.to(pt_model.device))
    
    # Check the shape of the soft prompt
    if pt_model.get_soft_prompt().shape != (pt_len, pt_model.config.hidden_size):
        print(f"Shape of the soft prompt is incorrect: {pt_model.get_soft_prompt().shape}")
        return False
    
    outputs_pt = pt_model.forward(input_ids=bg_token_ids.clone().to(pt_model.device))
    base_token_ids = bg_token_ids.clone().to(base_model.device)
    base_token_ids[:, pt_start:pt_end] = pt_token_ids
    outputs_base = base_model.forward(input_ids=base_token_ids)
    
    # Check the outputs of the prompt tuning model and the base model are the same
    if not torch.allclose(outputs_pt.logits, outputs_base.logits):
        print(f"Outputs of the prompt tuning model and the base model are not the same: {outputs_pt.logits} != {outputs_base.logits}")
        return False
    
    return True
    

def main():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    prompt_tuning_ranges = [
        (0, 10),
        (0, 20),
        (5, 20),
        (11, 37),
        (3, 200)
    ]
    pt_model, _ = load_model(model_name, (0, 10))
    base_model = LlamaForCausalLM.from_pretrained(model_name)
    base_model.cuda()
    
    # Testing init_soft_prompt_with_zero
    print("-"*100)
    print("Testing init_soft_prompt_with_zero")
    for prompt_tuning_range in prompt_tuning_ranges:
        pt_model.reset_soft_prompt(prompt_tuning_range)
        res =  test_init_soft_prompt_with_zero(pt_model)
        print(f"Prompt tuning range: {prompt_tuning_range}, Result: {res}")
    
    # Testing init_soft_prompt_with_random_values
    print("-"*100)
    print("Testing init_soft_prompt_with_random_values")
    for prompt_tuning_range in prompt_tuning_ranges:
        pt_model.reset_soft_prompt(prompt_tuning_range)
        res =  test_init_soft_prompt_with_random_values(pt_model)
        print(f"Prompt tuning range: {prompt_tuning_range}, Result: {res}")
    
    # Testing init_soft_prompt_with_prompt_embedding
    print("-"*100)
    print("Testing init_soft_prompt_with_prompt_embedding")
    for prompt_tuning_range in prompt_tuning_ranges:
        pt_model.reset_soft_prompt(prompt_tuning_range)
        res =  test_init_soft_prompt_with_prompt_embedding(pt_model, base_model)
        print(f"Prompt tuning range: {prompt_tuning_range}, Result: {res}")

if __name__ == "__main__":
    main()