from transformers import LlamaForCausalLM, LlamaConfig

from typing import Tuple, Optional

import torch
import torch.nn as nn


class LlamaPromptTuningConfig(LlamaConfig):
    def __init__(
        self,
        prompt_tuning_range: Optional[Tuple[int, int]] = None,
        *args,
        **kwargs,
    ):
        """
        prompt_tuning_range: The range of the soft prompt. E.g. (0, 10) means the first 10 tokens are the soft prompt.
        """
        super().__init__(*args, **kwargs)
        self.prompt_tuning_range = prompt_tuning_range


class LlamaPromptTuningLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        if config.prompt_tuning_range is not None:
            soft_prompt_len = config.prompt_tuning_range[1] - config.prompt_tuning_range[0]
            self.soft_prompt = nn.Parameter(torch.zeros(soft_prompt_len, config.hidden_size))
        else:
            self.soft_prompt = None
    
    def get_soft_prompt(self):
        return self.soft_prompt
    
    def reset_soft_prompt(
        self, 
        prompt_tuning_range: Tuple[int, int],
    ):
        print(f"Resetting soft prompt. You might want to initialize the soft prompt again.")
        self.config.prompt_tuning_range = prompt_tuning_range
        self.soft_prompt = nn.Parameter(torch.zeros(prompt_tuning_range[1] - prompt_tuning_range[0], self.config.hidden_size, device=self.device))
    
    def init_soft_prompt_with_zero(self):
        assert self.soft_prompt is not None, "Soft prompt is not initialized"
        self.soft_prompt.data.zero_()
    
    def init_soft_prompt_with_random_values(
        self,
        mean: float = 0.0,
        std: float = 0.02,
    ):
        assert self.soft_prompt is not None, "Soft prompt is not initialized"
        self.soft_prompt.data.normal_(mean, std)
    
    def init_soft_prompt_with_prompt_embedding(
        self,
        token_ids: torch.Tensor,
    ):
        """
        token_ids: int torch.Tensor with shape (seq_len)
        """
        assert self.soft_prompt is not None, "Soft prompt is not initialized"
        # Sanity check
        assert token_ids.ndim == 1, "token_ids must be a 1D tensor"
        # Check the length of the token ids
        soft_prompt_len = self.config.prompt_tuning_range[1] - self.config.prompt_tuning_range[0]
        assert token_ids.size(0) == soft_prompt_len, "The length of the token ids must be equal to the length of the soft prompt"
        
        # Get the prompt embedding
        prompt_embedding = self.model.embed_tokens(token_ids)
        self.soft_prompt.data.copy_(prompt_embedding)
        

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.config.prompt_tuning_range is not None:
            assert self.soft_prompt is not None, "Soft prompt is not initialized"
        
        if self.config.prompt_tuning_range is not None:
            # Get the input embeddings
            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)
            
            # Patch the soft prompt to the input
            start_idx = self.config.prompt_tuning_range[0]
            end_idx = self.config.prompt_tuning_range[1]
            inputs_embeds[:, start_idx:end_idx, :] = self.soft_prompt
                                
            return super().forward(inputs_embeds=inputs_embeds, **kwargs)
        else:
            # No prompt tuning
            return super().forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
