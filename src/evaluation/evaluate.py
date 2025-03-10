import json

def compute_accuracy(results):
    """Compute accuracy as the proportion of correct answers."""
    correct = sum(1 for r in results if r["model_output"][0] == r["reference_answer"])
    return correct / len(results)

def compute_exact_match(results):
    """Compute exact match score."""
    matches = sum(1 for r in results if any(output == r["reference_answer"] for output in r["model_output"]))
    return matches / len(results)

def compute_f1(results):
    """Compute F1-score for classification or QA tasks."""
    def f1_score(pred, gold):
        pred_tokens = set(pred.split())
        gold_tokens = set(gold.split())
        common = pred_tokens & gold_tokens
        if not common:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        return 2 * (precision * recall) / (precision + recall)
    
    f1_scores = [max(f1_score(output, r["reference_answer"]) for output in r["model_output"]) for r in results]
    return sum(f1_scores) / len(f1_scores)

def compute_pass_at_k(results, k):
    """Compute pass@k for code generation tasks."""
    passes = sum(1 for r in results if any(output == r["reference_answer"] for output in r["model_output"][:k]))
    return passes / len(results)

def load_json(filename):
    """Load JSON file with results."""
    with open(filename, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    results = load_json("results.json")
    print("Accuracy:", compute_accuracy(results))
    print("Exact Match:", compute_exact_match(results))
    print("F1 Score:", compute_f1(results))
    print("Pass@1:", compute_pass_at_k(results, 1))
    print("Pass@10:", compute_pass_at_k(results, 10))
    print("Pass@100:", compute_pass_at_k(results, 100))
