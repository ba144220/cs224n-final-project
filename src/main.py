from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import lovely_tensors

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


