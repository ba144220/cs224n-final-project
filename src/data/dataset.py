from datasets import load_dataset

def format_superglue(sample, task):
    """Format SuperGLUE dataset samples into QA format."""
    if task == "boolq":
        question = sample["question"]
        answer = sample["label"]
    elif task == "cb":
        question = f"Premise: {sample['premise']} Hypothesis: {sample['hypothesis']}"
        answer = sample["label"]  # 0: entailment, 1: contradiction, 2: neutral
    elif task == "copa":
        question = f"Premise: {sample['premise']} What is more plausible? {sample['choice1']} or {sample['choice2']}"
        answer = sample[f"choice{sample['label'] + 1}"]
    elif task == "multirc":
        question = f"Passage: {sample['paragraph']} Question: {sample['question']}"
        answer = "yes" if sample["answer"] else "no"
    elif task == "record":
        question = f"Passage: {sample['passage']} Question: {sample['query']}"
        answer = sample["answers"]
    elif task == "rte":
        question = f"Premise: {sample['premise']} Hypothesis: {sample['hypothesis']}"
        answer = "yes" if sample["label"] else "no"
    elif task == "wic":
        question = f"Sentence 1: {sample['sentence1']} Sentence 2: {sample['sentence2']} Word: {sample['word']}"
        answer = "yes" if sample["label"] else "no"
    elif task == "wsc":
        question = f"Sentence: {sample['text']} Does '{sample['span1_text']}' refer to '{sample['span2_text']}'?"
        answer = "yes" if sample["label"] else "no"
    else:
        question, answer = None, None
    return {"question": question, "answer": answer}

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
    dataset_qa["gsm8k"] = gsm8k["train"].map(lambda x: {"question": x["question"], "answer": x["answer"]})
    
    # MBPP
    mbpp = load_dataset("mbpp")
    dataset_qa["mbpp"] = mbpp["train"].map(lambda x: {"question": x["text"], "answer": x["code"]})
    
    
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
