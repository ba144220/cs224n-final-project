from transformers import LlamaForCausalLM, LlamaConfig

from typing import Tuple, Optional

import torch
import torch.nn as nn

import os

class LlamaPromptTuningLM(LlamaForCausalLM):
    """
    A class that extends LlamaForCausalLM to support prompt tuning.
    """
    
    soft_prompt_offset: Optional[int] = 4
    
    def __init__(self, config: LlamaConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
                
        # Freeze all the parameters
        for param in self.parameters():
            param.requires_grad = False
        
        self.soft_prompt = None
    def save_soft_prompt(self, output_dir: str ):
        if self.soft_prompt is None:
            raise ValueError("Soft prompt is not initialized")
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.soft_prompt.data, os.path.join(output_dir, "soft_prompt.pt"))
        
        print(f"Soft prompt saved to {output_dir}/soft_prompt.pt")
    
    def get_soft_prompt(self):
        return self.soft_prompt

    def get_soft_prompt_len(self):
        if self.soft_prompt is None:
           return 0
        return self.soft_prompt.shape[0]
    
    def load_soft_prompt(
        self,
        soft_prompt_path: str,
    ):
        self.soft_prompt = nn.Parameter(torch.load(soft_prompt_path))
    
    def init_soft_prompt_with_random_values(
        self,
        soft_prompt_len: int,
        random_range: float = 0.5,
    ):
        # Create the soft prompt
        self.soft_prompt = nn.Parameter(torch.zeros(soft_prompt_len, self.config.hidden_size, device=self.device))
        # Initialize the soft prompt with random values
        self.soft_prompt.data.uniform_(-random_range, random_range)
    
    def init_soft_prompt_with_prompt_embedding(
        self,
        token_ids: torch.Tensor,
    ):
        """
        token_ids: int torch.Tensor with shape (seq_len)
        """
        # Sanity check
        assert token_ids.ndim == 1, "token_ids must be a 1D tensor"
        # Get the length of the token ids
        soft_prompt_len = token_ids.size(0)
        self.soft_prompt = nn.Parameter(torch.zeros(soft_prompt_len, self.config.hidden_size, device=self.device))
        
        # Get the prompt embedding
        prompt_embedding = self.model.embed_tokens(token_ids)
        self.soft_prompt.data.copy_(prompt_embedding)
        

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        if self.soft_prompt is not None:
            # Get the input embeddings
            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)
            
            # Print the sum of attention mask
            if inputs_embeds.size(1) > 1: # if the input is not a single token
                # Patch the soft prompt to the input
                start_idx = self.soft_prompt_offset
                end_idx = start_idx + self.get_soft_prompt_len()
                # Expand self.soft_prompt to batch_size (copy itself batch_size times)
                expanded_soft_prompt = self.soft_prompt.unsqueeze(0).expand(inputs_embeds.size(0), -1, -1)
                inputs_embeds[:, start_idx:end_idx, :] = expanded_soft_prompt
                
            return super().forward(inputs_embeds=inputs_embeds, **kwargs)
        else:
            # No prompt tuning
            return super().forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
