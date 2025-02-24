from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_process_superglue(task, tokenizer):
    # Load the dataset for the specified task
    dataset = load_dataset("super_glue", task)
    
    # Process each split (train, validation, test) for tokenization
    def process_example(example):
        encoding = tokenizer(
            example["question"] if "question" in example else example["sentence"],  # for tasks like BoolQ or RTE
            example["passage"] if "passage" in example else None,  # for tasks that have a passage
            truncation=True,
            padding="max_length", 
            max_length=512,  
            return_tensors="pt"
        )
        # Depending on the task, add the label for classification (if available)
        if "label" in example:
            encoding["labels"] = example["label"]
        return encoding

    # Process the train, validation, and test splits
    dataset = dataset.map(process_example, batched=True)

    return dataset
