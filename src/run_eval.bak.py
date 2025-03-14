from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from models.llama_prompt_tuning import LlamaPromptTuningConfig, LlamaPromptTuningLM
from tqdm import tqdm


def format_prompt(context, question):
    """Format the input as a chat prompt"""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based on the provided context. You can only answer with Yes or No."
        },
        {   
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {question}"
        }
    ]
    return messages

def evaluate_model(
    model: LlamaPromptTuningLM,
    tokenizer: AutoTokenizer,
    num_examples: int = None,
    batch_size: int = 4
):
    """Evaluate the model on BoolQ validation set"""
    dataset = load_dataset("super_glue", "boolq")
    eval_dataset = dataset['validation']
    
    if num_examples:
        eval_dataset = eval_dataset.select(range(num_examples))
    
    # Preprocess the entire dataset at once
    def format_examples(examples):
        messages_list = [
            format_prompt(passage, question) 
            for passage, question in zip(examples['passage'], examples['question'])
        ]
        
        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ) for messages in messages_list
        ]
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'label': examples['label']
        }

    # Process dataset
    eval_dataset = eval_dataset.map(
        format_examples,
        batched=True,
        batch_size=100,
        remove_columns=eval_dataset.column_names
    )

    correct = 0
    unknown = 0
    total = 0
    
    # Evaluate in batches
    for i in tqdm(range(0, len(eval_dataset), batch_size)):
        batch_data = eval_dataset[i:i + batch_size]
        
        # Convert to tensors
        input_ids = torch.tensor(batch_data['input_ids']).to(model.device)
        attention_mask = torch.tensor(batch_data['attention_mask']).to(model.device)
        labels = batch_data['label']
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=5,
                do_sample=False
            )
        
        # Process predictions
        for j, output in enumerate(outputs):
            response = tokenizer.decode(output, skip_special_tokens=True)
            response = response.split("assistant")[-1].strip().lower()
            print(response)
            if "yes" in response:
                predicted = 1
            elif "no" in response:
                predicted = 0
            else:
                predicted = -1
                unknown += 1
            correct += (predicted == labels[j])
            total += 1
    
    accuracy = correct / total
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "unknown": unknown
    }

if __name__ == "__main__":
    # MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    MODEL_NAME = "output/random_init/checkpoint-500"
    TOKENIZER_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    
    # Load model and tokenizer
    config = LlamaPromptTuningConfig.from_pretrained(MODEL_NAME)
    # config.prompt_tuning_range = (5, 48)
    # config.prompt_tuning_range = (45, 48)
    
    model: LlamaPromptTuningLM = LlamaPromptTuningLM.from_pretrained(
        MODEL_NAME,  # Path to your trained model
        config=config,
        device_map="auto"
    )
    # model.init_soft_prompt_with_prompt_embedding(torch.tensor([271], device=model.device))
    # model.init_soft_prompt_with_random_values()
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Evaluate on a subset of 100 examples
    results = evaluate_model(model, tokenizer, batch_size=1, num_examples=400)
    
    print("\nEvaluation Results:")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Correct: {results['correct']}/{results['total']}")
    print(f"Unknown: {results['unknown']}/{results['total']}")
