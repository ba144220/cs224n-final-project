from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import lovely_tensors
from data.superglue import load_and_process_superglue
from data.gsm8k import load_and_process_gsm8k

lovely_tensors.monkey_patch()


model_name = "meta-llama/Llama-3.2-1B-Instruct"

print(f"Loading tokenizer and model from {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)
model.cuda()
print(f"{model = }")
print(f"{model.model.embed_tokens = }")
print(f"{model.model.embed_tokens.weight[:,0] = }")

# Print out the embedding


# Load and process example datasets
task = "boolq"
print(f"Processing SuperGLUE dataset: {task}")
dataset = load_and_process_superglue(task, tokenizer)

task = "gsm8k"
print(f"Processing dataset: {task}")
gsm8k_dataset = load_and_process_gsm8k(tokenizer)

# Print out a sample
print(f"Processed GSM8K example: {gsm8k_dataset['train'][0]}")
print(f"Processed dataset example: {dataset['train'][0]}")
