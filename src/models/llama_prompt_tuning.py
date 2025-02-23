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
            # TODO: Initialize the soft prompt with random values
            self.soft_prompt = nn.Parameter(torch.zeros(soft_prompt_len, config.hidden_size))
        else:
            self.soft_prompt = None

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.soft_prompt is not None and self.config.prompt_tuning_range is not None:
            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)
            
            # Patch the soft prompt to the input
            start_idx = self.config.prompt_tuning_range[0]
            end_idx = self.config.prompt_tuning_range[1]
            inputs_embeds[:, start_idx:end_idx, :] = self.soft_prompt
                                
            return super().forward(inputs_embeds=inputs_embeds, **kwargs)
        else:
            return super().forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
