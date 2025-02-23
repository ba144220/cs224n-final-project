import sys
import os

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_dir)

from transformers import LlamaForCausalLM
import torch
from typing import Tuple
from models.llama_prompt_tuning import LlamaPromptTuningConfig, LlamaPromptTuningLM

def test(
    model_base: LlamaForCausalLM,
    inputs_embeds: torch.Tensor,
):
    seq_len = inputs_embeds.shape[1]
    # Randomly select a start and end of the sequence length
    start = torch.randint(0, seq_len - 2, (1,))
    end = torch.randint(start + 1, seq_len, (1,))
    
    start = 1
    end = 2
    
    # Initialize the model
    config = LlamaPromptTuningConfig.from_pretrained(MODEL_NAME)
    config.prompt_tuning_range = (start, end)
    if "model_pt" in locals():
        del model_pt
        torch.cuda.empty_cache()
    
    model_pt = LlamaPromptTuningLM.from_pretrained(MODEL_NAME, config=config)
    model_pt.cuda()
    
    inputs_embeds_original = inputs_embeds.clone()
    inputs_embeds_patched = inputs_embeds.clone()
    
    # We zero out the prompt tuning range
    inputs_embeds_patched[:, start:end, :] = torch.zeros_like(inputs_embeds_patched[:, start:end, :])
    
    outputs_pt = model_pt(inputs_embeds=inputs_embeds_original, return_dict=True)
    outputs_base = model_base(inputs_embeds=inputs_embeds_patched, return_dict=True)
    
    return torch.allclose(outputs_pt.logits, outputs_base.logits)


if __name__ == "__main__":
    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    config = LlamaPromptTuningConfig.from_pretrained(MODEL_NAME)
    config.prompt_tuning_range = (1, 2)
    
    model_base = LlamaForCausalLM.from_pretrained(MODEL_NAME)
    model_base.cuda()
    
    for i in range(10):
        print(f"Test {i}")
        inputs_embeds = torch.randn(2, 40, config.hidden_size) # (batch_size, sequence_length, hidden_size)
        inputs_embeds = inputs_embeds.to(model_base.device)
        
        result = test( model_base, inputs_embeds)
        print(f"{result = }")
        if not result:
            raise Exception("Test failed")
        
    print("Test passed")