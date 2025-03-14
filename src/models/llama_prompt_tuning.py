from transformers import LlamaForCausalLM, LlamaConfig

from typing import Optional, Union, List

import torch
import torch.nn.functional as F
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
        soft_prompt_len: Optional[int] = None,
    ):
        """
        token_ids: int torch.Tensor with shape (seq_len)
        """
        # Sanity check
        assert token_ids.ndim == 1, "token_ids must be a 1D tensor"
        
        token_ids_len = token_ids.size(0)
        if soft_prompt_len is not None:
            print(f"\nSoft prompt length specified. Will do interpolation.\n")
        else:
            soft_prompt_len = token_ids_len
        self.soft_prompt = nn.Parameter(torch.zeros(soft_prompt_len, self.config.hidden_size, device=self.device))
        
        # Get the prompt embedding
        prompt_embedding = self.model.embed_tokens(token_ids)
        if token_ids_len != soft_prompt_len:
            print(f"Interpolating prompt embedding from {token_ids_len} to {soft_prompt_len} tokens")
            # Unsqueeze the prompt embedding to (1, 1, seq_len, hidden_size)
            prompt_embedding = prompt_embedding.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, hidden_size)
            prompt_embedding = F.interpolate(
                prompt_embedding, 
                size=(soft_prompt_len, prompt_embedding.size(1)),
                mode="bilinear", 
                align_corners=False
            )
            prompt_embedding = prompt_embedding.squeeze(0).squeeze(0) # (seq_len, hidden_size)
        self.soft_prompt.data.copy_(prompt_embedding)
        

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        padding_offsets: Optional[Union[int, List[int]]] = 0,
        **kwargs,
    ) -> torch.Tensor:

        if self.soft_prompt is not None:
            
            # Get the input embeddings
            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)
                
            # Get batch size
            batch_size = inputs_embeds.size(0)
            
            if isinstance(padding_offsets, int):
                padding_offsets = [padding_offsets] * batch_size
            else:
                assert len(padding_offsets) == batch_size, "padding_offsets must be a list of length batch_size"
            
            # Print the sum of attention mask
            if inputs_embeds.size(1) > 1: # if the input is not a single token
                for batch_idx in range(batch_size):
                    # Patch the soft prompt to the input
                    start_idx = self.soft_prompt_offset + padding_offsets[batch_idx]
                    end_idx = start_idx + self.get_soft_prompt_len()
                    inputs_embeds[batch_idx, start_idx:end_idx, :] = self.soft_prompt
                
            return super().forward(inputs_embeds=inputs_embeds, **kwargs)
        else:
            # No prompt tuning
            return super().forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
