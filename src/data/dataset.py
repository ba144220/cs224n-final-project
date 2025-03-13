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
    # HUMAN_EVAL = "humaneval"
    
default_system_prompts = {
    DatasetEnum.BOOLQ: (
        "You are given a passage and a question. Read the passage carefully and answer with a single word: "
        "either \"yes\" or \"no\". Provide no additional text."
    ),
    DatasetEnum.CB: (
        "You are given a premise and a hypothesis. Determine the logical relationship between them. "
        "Answer with one of the following single words: \"entailment\", \"contradiction\", or \"neutral\". "
        "Provide no additional text."
    ),
    DatasetEnum.COPA: (
        "You are given a premise and two options labeled \"option1\" and \"option2\". Based on the premise, "
        "decide which option is more plausible. Output only the label (\"option1\" or \"option2\") with no additional text."
    ),
    DatasetEnum.MULTIRC: (
        "You are given a passage and a question. Analyze the passage and answer with a single word: "
        "either \"yes\" or \"no\". Provide no additional text."
    ),
    DatasetEnum.RECORD: (
        "You are given a passage and a question. Identify all correct answers based on the passage. "
        "Output your answer as a comma-separated list of answer phrases with no additional text."
    ),
    DatasetEnum.RTE: (
        "You are given a premise and a hypothesis. Evaluate whether the hypothesis logically follows from the premise. "
        "Answer with a single word: \"yes\" if it does, or \"no\" if it does not. Provide no additional text."
    ),
    DatasetEnum.WIC: (
        "You are given two sentences and a target word. Determine whether the target word is used in the same sense in both sentences. "
        "Answer with a single word: \"yes\" if it is, or \"no\" if it is not. Provide no additional text."
    ),
    DatasetEnum.WSC: (
        "You are given a sentence with two highlighted spans. Decide whether the first span refers to the second span. "
        "Answer with a single word: \"yes\" if it does, or \"no\" if it does not. Provide no additional text."
    ),
    DatasetEnum.GSM8K: (
        "You are given a math word problem. Read the problem carefully and work through it step-by-step to reach a solution. "
        "Show all intermediate calculations and reasoning in a clear chain-of-thought. "
        "Conclude your response with your final answer on a new line starting with '#### ' followed immediately by the numerical answer. "
        "Provide no additional text beyond the reasoning and the final answer."
    ),
    DatasetEnum.MBPP: (
        "You are given a programming task described in the input. Write a Python function that implements the solution as specified. "
        "Output only the Python code with no additional commentary, explanations, or extraneous text."
    ),
    # DatasetEnum.HUMAN_EVAL: (
    #     "You are given a programming task described in the input. Write a Python function that implements the solution as specified. "
    #     "Output only the Python code with no additional commentary, explanations, or extraneous text."
    # )
}

def format_dataset(sample, dataset_name: DatasetEnum):
    """Format dataset samples into a standardized QA format."""
    
    # SuperGLUE
    if dataset_name == DatasetEnum.BOOLQ:
        input_text = f"# Passage:\n{sample['passage']}\n# Question:\n{sample['question']}"
        answer = "yes" if sample["label"] else "no"
    elif dataset_name == DatasetEnum.CB:
        input_text = f"# Premise:\n{sample['premise']}\n# Hypothesis:\n{sample['hypothesis']}"
        answer = ["entailment", "contradiction", "neutral"][sample["label"]]
    elif dataset_name == DatasetEnum.COPA:
        input_text = f"# Premise:\n{sample['premise']}\n# Question:\nWhat is more plausible?\n# Choice 1:\n{sample['choice1']}\n# Choice 2:\n{sample['choice2']}"
        answer = sample[f"choice{sample['label'] + 1}"]
    elif dataset_name == DatasetEnum.MULTIRC:
        input_text = f"# Passage:\n{sample['paragraph']}\n# Question:\n{sample['question']}"
        answer = "yes" if sample["answer"] else "no"
    elif dataset_name == DatasetEnum.RECORD:
        input_text = f"# Passage:\n{sample['passage']}\n# Question:\n{sample['query']}"
        answer = ", ".join(sample["answers"])
    elif dataset_name == DatasetEnum.RTE:
        input_text = f"# Premise:\n{sample['premise']}\n# Hypothesis:\n{sample['hypothesis']}"
        answer = "yes" if sample["label"] else "no"
    elif dataset_name == DatasetEnum.WIC:
        input_text = f"# Sentence 1:\n{sample['sentence1']}\n# Sentence 2:\n{sample['sentence2']}\n# Word:\n{sample['word']}"
        answer = "yes" if sample["label"] else "no"
    elif dataset_name == DatasetEnum.WSC:
        input_text = f"# Sentence:\n{sample['text']}\n# Does '{sample['span1_text']}' refer to '{sample['span2_text']}'?"
        answer = "yes" if sample["label"] else "no"
    # GSM8K
    elif dataset_name == DatasetEnum.GSM8K:
        input_text = f"# Question:\n{sample['question']}"
        answer = sample["answer"]
    # MBPP
    elif dataset_name == DatasetEnum.MBPP:
        input_text = f"# Task:\n{sample['text']}"
        answer = sample["code"]
    # # HumanEval
    # elif dataset_name == DatasetEnum.HUMAN_EVAL:
    #     input_text = f"# Prompt:\n{sample['prompt']}"
    #     answer = sample["canonical_solution"]
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
    
    
    print(DatasetEnum.GSM8K)
    dataset = load_pt_dataset(DatasetEnum.GSM8K)
    print(dataset)
    print(dataset["test"][0])
    
    print(DatasetEnum.MBPP)
    dataset = load_pt_dataset(DatasetEnum.MBPP)
    print(dataset)
    print(dataset["test"][0])
    