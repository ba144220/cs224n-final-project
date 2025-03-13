from dataclasses import dataclass, field
from typing import Optional

from data.dataset import DatasetEnum

@dataclass
class ModelArguments:
    model_name: str
    
    # Prompt tuning parameters
    use_prompt_tuning: bool
    
    # Initial soft prompt
    init_from_natural_language: Optional[bool] = field(default=False)
    # Path to a pretrained model to initialize the soft prompt from
    init_from_pretrained: Optional[str] = field(default=None)
    init_random: Optional[bool] = field(default=False)
    
    # System prompt
    # If init_from_natural_language is True, this will be used to initialize the soft prompt
    # If use_prompt_tuning is False, this will be the system prompt
    system_prompt: Optional[str] = field(default=None)
    
    # Prompt tuning length
    # If init_random is True, this will be the length of the random prompt
    # If init_from_natural_language is True
    # or init_from_pretrained is True 
    # or use_prompt_tuning is False, this will be ignored
    prompt_tuning_length: int

@dataclass
class DatasetArguments:
    dataset: DatasetEnum = field(
        metadata={"help": "The name of the dataset to use. Available options: " + ", ".join([e.value for e in DatasetEnum])}
    )
    train_size: Optional[int] = field(default=None)
    eval_size: Optional[int] = field(default=None)
