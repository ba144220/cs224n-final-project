import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def evaluate_boolq(model_path, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
    dataset = load_dataset("super_glue", "boolq")["validation"]
    
    correct = 0
    total = 0

    for sample in tqdm(dataset):
        question = sample["question"]
        passage = sample["passage"]
        label = sample["answer"]  # 1 for True, 0 for False

        # Format input
        input_text = f"Passage: {passage}\nQuestion: {question}\nAnswer: "
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        # Generate response
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=5)

        response = tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()
        
        if "yes" in response:
            prediction = 1
        elif "no" in response:
            prediction = 0
        else:
            prediction = -1  # Invalid prediction
        
        if prediction == label:
            correct += 1
        total += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy

# Example usage:
# evaluate_boolq("/path/to/your/saved/model")
