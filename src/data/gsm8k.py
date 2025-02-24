from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_process_gsm8k(tokenizer):
    dataset = load_dataset("gsm8k", "main")
    
    # Process the dataset with tokenization
    def process_example(example):
        encoding = tokenizer(
            example["question"],  
            truncation=True,
            padding="max_length", 
            max_length=512,  
            return_tensors="pt"
        )
        
        if "answer" in example:
            encoding["labels"] = example["answer"]
        return encoding
    
    dataset = dataset.map(process_example, batched=True)
    return dataset
