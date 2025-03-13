from datasets import load_dataset
from enum import Enum

class DatasetEnum(Enum):
    # SuperGLUE
    BOOLQ = "boolq"
    CB = "cb"
    COPA = "copa"
    MULTIRC = "multirc"
    RECORD = "record"
    RTE = "rte"
    WIC = "wic"
    WSC = "wsc"
    # Other
    GSM8K = "gsm8k"
    MBPP = "mbpp"
    HUMAN_EVAL = "humaneval"

def format_dataset(sample, dataset_name: DatasetEnum):
    """Format dataset samples into a standardized QA format."""
    
    # SuperGLUE
    if dataset_name == DatasetEnum.BOOLQ:
        input_text = f"Passage: {sample['passage']}\nQuestion: {sample['question']}"
        answer = "yes" if sample["label"] else "no"
    elif dataset_name == DatasetEnum.CB:
        input_text = f"Premise: {sample['premise']}\nHypothesis: {sample['hypothesis']}"
        answer = ["entailment", "contradiction", "neutral"][sample["label"]]
    elif dataset_name == DatasetEnum.COPA:
        input_text = f"Premise: {sample['premise']}\nWhat is more plausible? {sample['choice1']} or {sample['choice2']}?"
        answer = sample[f"choice{sample['label'] + 1}"]
    elif dataset_name == DatasetEnum.MULTIRC:
        input_text = f"Passage: {sample['paragraph']}\nQuestion: {sample['question']}"
        answer = "yes" if sample["answer"] else "no"
    elif dataset_name == DatasetEnum.RECORD:
        input_text = f"Passage: {sample['passage']}\nQuestion: {sample['query']}"
        answer = ", ".join(sample["answers"])
    elif dataset_name == DatasetEnum.RTE:
        input_text = f"Premise: {sample['premise']}\nHypothesis: {sample['hypothesis']}"
        answer = "yes" if sample["label"] else "no"
    elif dataset_name == DatasetEnum.WIC:
        input_text = f"Sentence 1: {sample['sentence1']}\nSentence 2: {sample['sentence2']}\nWord: {sample['word']}"
        answer = "yes" if sample["label"] else "no"
    elif dataset_name == DatasetEnum.WSC:
        input_text = f"Sentence: {sample['text']}\nDoes '{sample['span1_text']}' refer to '{sample['span2_text']}'?"
        answer = "yes" if sample["label"] else "no"
    # GSM8K
    elif dataset_name == DatasetEnum.GSM8K:
        input_text = f"Question: {sample['question']}"
        answer = sample["answer"]
    # MBPP
    elif dataset_name == DatasetEnum.MBPP:
        input_text = f"Task: {sample['text']}"
        answer = sample["code"]
    # HumanEval
    elif dataset_name == DatasetEnum.HUMAN_EVAL:
        input_text = f"Prompt: {sample['prompt']}"
        answer = sample["canonical_solution"]
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return {"input_text": input_text, "answer": answer}

def load_pt_dataset(dataset_name: DatasetEnum):
    """Load and format a specific dataset."""
    
    superglue_tasks = [
        DatasetEnum.BOOLQ, 
        DatasetEnum.CB, 
        DatasetEnum.COPA, 
        DatasetEnum.MULTIRC, 
        DatasetEnum.RECORD, 
        DatasetEnum.RTE, 
        DatasetEnum.WIC, 
        DatasetEnum.WSC
    ]
    if dataset_name in superglue_tasks:
        dataset = load_dataset("super_glue", dataset_name.value)
        # Remove test split and replace with validation split
        dataset["test"] = dataset["validation"]
        train_dataset = dataset["train"].train_test_split(test_size=0.1)
        dataset["train"] = train_dataset["train"]
        dataset["validation"] = train_dataset["test"]
        return dataset.map(lambda x: format_dataset(x, dataset_name))
    elif dataset_name == DatasetEnum.GSM8K:
        dataset = load_dataset("gsm8k", "main")
        dataset = dataset.map(lambda x: format_dataset(x, dataset_name))
        # Split 10% of train split as validation set
        train_dataset = dataset["train"].train_test_split(test_size=0.1)
        dataset["train"] = train_dataset["train"]
        dataset["validation"] = train_dataset["test"]
        return dataset
    elif dataset_name == DatasetEnum.MBPP:
        dataset = load_dataset("mbpp")
        dataset = dataset.map(lambda x: format_dataset(x, dataset_name))
        return dataset
    elif dataset_name == DatasetEnum.HUMAN_EVAL:
        dataset = load_dataset("openai/openai_humaneval")
        return dataset.map(lambda x: format_dataset(x, dataset_name))
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
if __name__ == "__main__":
    print(DatasetEnum.CB)
    dataset = load_pt_dataset(DatasetEnum.CB)
    print(dataset)
    print(dataset["test"][0])
    
    print(DatasetEnum.HUMAN_EVAL)
    dataset = load_pt_dataset(DatasetEnum.HUMAN_EVAL)
    print(dataset)
    print(dataset["test"][0])
    
    print(DatasetEnum.GSM8K)
    dataset = load_pt_dataset(DatasetEnum.GSM8K)
    print(dataset)
    print(dataset["test"][0])
    
    print(DatasetEnum.MBPP)
    dataset = load_pt_dataset(DatasetEnum.MBPP)
    print(dataset)
    print(dataset["test"][0])
    