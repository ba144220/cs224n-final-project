from datasets import load_dataset

def format_superglue(sample, task):
    """Format SuperGLUE dataset samples into a standardized QA format."""
    if task == "boolq":
        input_text = f"Passage: {sample['passage']}\nQuestion: {sample['question']}\nAnswer: "
        answer = "yes" if sample["label"] else "no"
    elif task == "cb":
        input_text = f"Premise: {sample['premise']}\nHypothesis: {sample['hypothesis']}\nAnswer: "
        answer = ["entailment", "contradiction", "neutral"][sample["label"]]
    elif task == "copa":
        input_text = f"Premise: {sample['premise']}\nWhat is more plausible? {sample['choice1']} or {sample['choice2']}?\nAnswer: "
        answer = sample[f"choice{sample['label'] + 1}"]
    elif task == "multirc":
        input_text = f"Passage: {sample['paragraph']}\nQuestion: {sample['question']}\nAnswer: "
        answer = "yes" if sample["answer"] else "no"
    elif task == "record":
        input_text = f"Passage: {sample['passage']}\nQuestion: {sample['query']}\nAnswer: "
        answer = ", ".join(sample["answers"])
    elif task == "rte":
        input_text = f"Premise: {sample['premise']}\nHypothesis: {sample['hypothesis']}\nAnswer: "
        answer = "yes" if sample["label"] else "no"
    elif task == "wic":
        input_text = f"Sentence 1: {sample['sentence1']}\nSentence 2: {sample['sentence2']}\nWord: {sample['word']}\nAnswer: "
        answer = "yes" if sample["label"] else "no"
    elif task == "wsc":
        input_text = f"Sentence: {sample['text']}\nDoes '{sample['span1_text']}' refer to '{sample['span2_text']}'?\nAnswer: "
        answer = "yes" if sample["label"] else "no"
    else:
        input_text, answer = None, None

    return {"input_text": input_text, "answer": answer}

def load_superglue():
    """Load and format SuperGLUE datasets."""
    superglue_tasks = ["boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc"]
    dataset_qa = {}
    for task in superglue_tasks:
        dataset = load_dataset("super_glue", task)
        dataset_qa[task] = dataset["train"].map(lambda x: format_superglue(x, task))
    return dataset_qa

def load_other_datasets():
    """Load and format other generative QA datasets."""
    dataset_qa = {}
    
    # GSM8K
    gsm8k = load_dataset("gsm8k", "main")
    dataset_qa["gsm8k"] = gsm8k["train"].map(lambda x: {
        "input_text": f"Question: {x['question']}\nAnswer: ",
        "answer": x["answer"]
    })
    
    # MBPP
    mbpp = load_dataset("mbpp")
    dataset_qa["mbpp"] = mbpp["train"].map(lambda x: {
        "input_text": f"Task: {x['text']}\nCode: ",
        "answer": x["code"]
    })
    
    # HumanEval
    humaneval = load_dataset("openai/openai_humaneval")
    dataset_qa["humaneval"] = humaneval["test"].map(lambda x: {
        "input_text": f"Prompt: {x['prompt']}\nSolution: ",
        "answer": x["canonical_solution"]
    })
    
    return dataset_qa

def load_all_datasets():
    """Load and format all datasets."""
    dataset_qa = load_superglue()
    dataset_qa.update(load_other_datasets())
    return dataset_qa

if __name__ == "__main__":
    dataset_qa = load_all_datasets()
    print(dataset_qa["boolq"][0])
    print(dataset_qa["gsm8k"][0])
    print(dataset_qa["mbpp"][0])
    print(dataset_qa["humaneval"][0])
