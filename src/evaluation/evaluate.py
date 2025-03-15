import json
import re
import subprocess
import tempfile
# from sklearn.metrics import f1_score

def load_json(filename):
    """Load JSON file with results."""
    with open(filename, "r") as f:
        return json.load(f)

def extract_last_number(text):
    numbers = re.findall(r'\d+', text)
    return int(numbers[-1]) if numbers else None

def evaluate_boolq(results):
    """Compute accuracy and incorrect format proportion for BoolQ dataset."""
    correct = 0
    incorrect_format = 0
    for r in results:
        model_output = r["model_output"].lower()
        if "yes" in model_output and "no" in model_output:
            incorrect_format += 1
        elif "yes" in model_output:
            if r["reference_answer"] == "yes":
                correct += 1
        elif "no" in model_output:
            if r["reference_answer"] == "no":
                correct += 1
        else:
            incorrect_format += 1
    accuracy = correct / len(results)
    incorrect_format_proportion = incorrect_format / len(results)
    return accuracy, incorrect_format_proportion

# def compute_f1(results):
#     """Compute F1 score for text classification tasks."""
#     y_true = [entry["reference_answer"].strip().lower() for entry in results]
#     y_pred = [entry["model_output"].strip().lower() for entry in results]
    
#     unique_labels = list(set(y_true))  # Ensure labels match reference classes
#     f1 = f1_score(y_true, y_pred, average='macro', labels=unique_labels)
    
#     return f1

def evaluate_gsm8k(results):
    """Compute accuracy for GSM8k dataset, comparing last numberical value in the output."""
    correct = sum(1 for r in results if extract_last_number(r["model_output"]) == extract_last_number(r["reference_answer"]))
    return (correct / len(results))

def evaluate_mbpp(results):
    """Computer Pass@1 for MBPP dataset."""
    passed = sum(1 for r in results if run_tests(r["model_output"], r["test_list"]))
    return (passed / len(results))

def run_tests(code, test_list, timeout=5):
    """Executes the model's code with the provided test cases and checks if they pass."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
        temp_file.write(code + "\n")
        for test in test_list:
            temp_file.write(test + "\n")
        temp_file_path = temp_file.name

    try:
        result = subprocess.run(
            ["python3", temp_file_path],
            capture_output=True,
            text=True,
            timeout=timeout  # Set a timeout to prevent infinite loops
        )
        return result.returncode == 0  # Return True if all tests pass, False otherwise
    except subprocess.TimeoutExpired:
        return False  # Return False if execution times out
    except Exception as e:
        return False

# def compute_pass_at_k(results, k):
#     """Compute pass@k for code generation tasks."""
#     passes = sum(1 for r in results if any(output == r["reference_answer"] for output in r["model_output"][:k]))
#     return passes / len(results)

# def compute_exact_match(results):
#     """Compute exact match score."""
#     matches = sum(1 for r in results if any(output == r["reference_answer"] for output in r["model_output"]))
#     return matches / len(results)


if __name__ == "__main__":
    results = load_json("./results/wic/wic_rand_32_eval_results.json")
    accuracy, incorrect_format_proportion = evaluate_boolq(results)
    print("Accuracy:", accuracy)
    print("Incorrect format:", incorrect_format_proportion)
    # print("Accuracy:", evaluate_gsm8k(results))
    # print("Pass@1:", evaluate_mbpp(results))

