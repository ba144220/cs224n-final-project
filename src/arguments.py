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
    # If it's None, the default system prompt will be used
    system_prompt: Optional[str] = field(default=None)
    
    # Prompt tuning length
    # If init_random is True, this will be the length of the random prompt
    # If init_from_natural_language is True
    # or init_from_pretrained is True 
    # or use_prompt_tuning is False, this will be ignored
    prompt_tuning_length: int = field(default=None)

@dataclass
class DatasetArguments:
    dataset: DatasetEnum = field(
        metadata={"help": "The name of the dataset to use. Available options: " + ", ".join([e.value for e in DatasetEnum])}
    )
    train_size: Optional[int] = field(default=None)
    eval_size: Optional[int] = field(default=None)


@dataclass
class TrainingArguments:
    output_dir: str = field(default="./outputs")
    gradient_accumulation_steps: int = field(default=1)
    num_train_epochs: int = field(default=1)
    max_steps: int = field(default=1000)
    
    logging_steps: int = field(default=10)
    max_seq_length: int = field(default=2048)
    
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    learning_rate: float = field(default=2.0e-5)
    weight_decay: float = field(default=0.01)
    warmup_steps: int = field(default=50)
    
    do_train: bool = field(default=True)
    do_eval: bool = field(default=True)
    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=50)
    eval_on_start: bool = field(default=True)
    
    save_strategy: str = field(default="steps")
    save_total_limit: int = field(default=1)
    save_steps: int = field(default=100)
