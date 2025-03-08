import sys
import os

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_dir)

from models.llama_prompt_tuning import LlamaPromptTuningConfig, LlamaPromptTuningLM

import lovely_tensors

lovely_tensors.monkey_patch()

SOFT_PROMPT_PATH = "./outputs/soft_prompt.pt"
OUTPUT_DIR = "./outputs"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
RANGE = (0, 10)

def test_save_soft_prompt():
    config = LlamaPromptTuningConfig.from_pretrained(MODEL_NAME)
    config.output_dir = OUTPUT_DIR
    config.prompt_tuning_range = RANGE
    
    model: LlamaPromptTuningLM = LlamaPromptTuningLM(config).cuda()
    model.init_soft_prompt_with_random_values()
    model.save_soft_prompt()
    
    print(f"Soft prompt saved to {OUTPUT_DIR}/soft_prompt.pt")

def test_load_soft_prompt():
    config = LlamaPromptTuningConfig.from_pretrained(MODEL_NAME)
    config.soft_prompt_path = SOFT_PROMPT_PATH
    config.prompt_tuning_range = RANGE
    
    model = LlamaPromptTuningLM(config).cuda()
    
    print(model.soft_prompt)


if __name__ == "__main__":
    test_save_soft_prompt()
    test_load_soft_prompt()
