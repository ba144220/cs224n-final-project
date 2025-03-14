from datasets import load_dataset
from enum import Enum


class DatasetEnum(Enum):
    # SuperGLUE
    BOOLQ = "boolq"
    CB = "cb"
    COPA = "copa"
    # MULTIRC = "multirc"
    # RECORD = "record"
    RTE = "rte"
    WIC = "wic"
    WSC = "wsc"
    # Other
    GSM8K = "gsm8k"
    MBPP = "mbpp"
    # HUMAN_EVAL = "humaneval"
    
generation_lengths = {
    DatasetEnum.BOOLQ.value: 5,
    DatasetEnum.CB.value: 5,
    DatasetEnum.COPA.value: 5,
    # DatasetEnum.MULTIRC.value: 5,
    # DatasetEnum.RECORD.value: 120,
    DatasetEnum.RTE.value: 5,
    DatasetEnum.WIC.value: 5,
    DatasetEnum.WSC.value: 5,
    DatasetEnum.GSM8K.value: 256,
    DatasetEnum.MBPP.value: 256,
}
    
default_system_prompts = {
    DatasetEnum.BOOLQ.value: (
        "You are given a passage and a question. Read the passage carefully and answer with a single word: "
        "either \"yes\" or \"no\". Provide no additional text."
    ),
    DatasetEnum.CB.value: (
        "You are given a premise and a hypothesis. Determine the logical relationship between them. "
        "Answer with one of the following single words: \"entailment\", \"contradiction\", or \"neutral\". "
        "Provide no additional text."
    ),
    DatasetEnum.COPA.value: (
        "You are given a premise and two options labeled \"option1\" and \"option2\". Based on the premise, "
        "decide which option is more plausible. Output only the label (\"option1\" or \"option2\") with no additional text."
    ),
    # DatasetEnum.MULTIRC.value: (
    #     "You are given a passage and a question. Analyze the passage and answer with a single word: "
    #     "either \"yes\" or \"no\". Provide no additional text."
    # ),
    # DatasetEnum.RECORD.value: (
    #     "You are given a passage and a question. Identify all correct answers based on the passage. "
    #     "Output your answer as a comma-separated list of answer phrases with no additional text."
    # ),
    DatasetEnum.RTE.value: (
        "You are given a premise and a hypothesis. Evaluate whether the hypothesis logically follows from the premise. "
        "Answer with a single word: \"yes\" if it does, or \"no\" if it does not. Provide no additional text."
    ),
    DatasetEnum.WIC.value: (
        "You are given two sentences and a target word. Determine whether the target word is used in the same sense in both sentences. "
        "Answer with a single word: \"yes\" if it is, or \"no\" if it is not. Provide no additional text."
    ),
    DatasetEnum.WSC.value: (
        "You are given a sentence with two highlighted spans. Decide whether the first span refers to the second span. "
        "Answer with a single word: \"yes\" if it does, or \"no\" if it does not. Provide no additional text."
    ),
    DatasetEnum.GSM8K.value: (
        "You are given a math word problem. Read the problem carefully and work through it step-by-step to reach a solution. "
        "Show all intermediate calculations and reasoning in a clear chain-of-thought. "
        "At the end, conclude your response with your final answer on a new line starting with 'Answer: ' followed immediately by the numerical answer. "
        "Provide no additional text beyond the reasoning and the final answer."
    ),
    DatasetEnum.MBPP.value: (
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
    if dataset_name == DatasetEnum.BOOLQ.value:
        input_text = f"# Passage:\n{sample['passage']}\n# Question:\n{sample['question']}"
        answer = "yes" if sample["label"] else "no"
    elif dataset_name == DatasetEnum.CB.value:
        input_text = f"# Premise:\n{sample['premise']}\n# Hypothesis:\n{sample['hypothesis']}"
        answer = ["entailment", "contradiction", "neutral"][sample["label"]]
    elif dataset_name == DatasetEnum.COPA.value:
        input_text = f"# Premise:\n{sample['premise']}\n# Question:\nWhat is more plausible?\noption1:\n{sample['choice1']}\noption2:\n{sample['choice2']}"
        # answer = sample[f"choice{sample['label'] + 1}"]
        answer = f"option{sample['label'] + 1}"
    # elif dataset_name == DatasetEnum.MULTIRC.value:
    #     input_text = f"# Passage:\n{sample['paragraph']}\n# Question:\n{sample['question']}"
    #     answer = "yes" if sample["answer"] else "no"
    # elif dataset_name == DatasetEnum.RECORD.value:
    #     input_text = f"# Passage:\n{sample['passage']}\n# Question:\n{sample['query']}"
    #     answer = ", ".join(sample["answers"])
    elif dataset_name == DatasetEnum.RTE.value:
        input_text = f"# Premise:\n{sample['premise']}\n# Hypothesis:\n{sample['hypothesis']}"
        answer = "yes" if sample["label"] else "no"
    elif dataset_name == DatasetEnum.WIC.value:
        input_text = f"# Sentence 1:\n{sample['sentence1']}\n# Sentence 2:\n{sample['sentence2']}\n# Word:\n{sample['word']}"
        answer = "yes" if sample["label"] else "no"
    elif dataset_name == DatasetEnum.WSC.value:
        input_text = f"# Sentence:\n{sample['text']}\n# Does '{sample['span1_text']}' refer to '{sample['span2_text']}'?"
        answer = "yes" if sample["label"] else "no"
    # GSM8K
    elif dataset_name == DatasetEnum.GSM8K.value:
        input_text = f"# Question:\n{sample['question']}"
        answer = sample["answer"].replace("#### ", "Answer: ")
        # print(answer)
    # MBPP
    elif dataset_name == DatasetEnum.MBPP.value:
        input_text = f"# Task:\n{sample['text']}\nHere is a test case for the function: {sample['test_list'][0]}"
        answer = sample["code"]
        # print(input_text)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return {"input_text": input_text, "answer": answer}

def load_pt_dataset(dataset_name: str):
    """Load and format a specific dataset."""
    
    superglue_tasks = [
        DatasetEnum.BOOLQ.value, 
        DatasetEnum.CB.value, 
        DatasetEnum.COPA.value, 
        # DatasetEnum.MULTIRC.value, 
        # DatasetEnum.RECORD.value, 
        DatasetEnum.RTE.value, 
        DatasetEnum.WIC.value, 
        DatasetEnum.WSC.value
    ]
    if dataset_name in superglue_tasks:
        dataset = load_dataset("super_glue", dataset_name)
        # Remove test split and replace with validation split
        dataset["test"] = dataset["validation"]
        train_dataset = dataset["train"].train_test_split(test_size=0.1)
        dataset["train"] = train_dataset["train"]
        dataset["validation"] = train_dataset["test"]
        return dataset.map(lambda x: format_dataset(x, dataset_name))
    elif dataset_name == DatasetEnum.GSM8K.value:
        dataset = load_dataset("gsm8k", "main")
        dataset = dataset.map(lambda x: format_dataset(x, dataset_name))
        # Split 10% of train split as validation set
        train_dataset = dataset["train"].train_test_split(test_size=0.1)
        dataset["train"] = train_dataset["train"]
        dataset["validation"] = train_dataset["test"]
        return dataset
    elif dataset_name == DatasetEnum.MBPP.value:
        dataset = load_dataset("mbpp")
        dataset = dataset.map(lambda x: format_dataset(x, dataset_name))
        return dataset
    # elif dataset_name == DatasetEnum.HUMAN_EVAL:
    #     dataset = load_dataset("openai/openai_humaneval")
    #     return dataset.map(lambda x: format_dataset(x, dataset_name))
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
if __name__ == "__main__":
    print(DatasetEnum.WIC.value)
    dataset = load_pt_dataset(DatasetEnum.WIC.value)
    print(dataset)
    for i in range(10):
        print(dataset["test"][i]["input_text"])
        print(dataset["test"][i]["answer"])
    
    # print(DatasetEnum.GSM8K.value)
    # dataset = load_pt_dataset(DatasetEnum.GSM8K.value)
    # print(dataset)
    # print(dataset["test"][0])
    
    # print(DatasetEnum.MBPP.value)
    # dataset = load_pt_dataset(DatasetEnum.MBPP.value)
    # print(dataset)
    # print(dataset["test"][0])
    