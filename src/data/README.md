Currently, we have the following datasets:
- SuperGLUE
    - BoolQ
    - CB
    - COPA
    - MULTIRC
    - RECORD
    - RTE
    - WIC
    - WSC
- GSM8K
- MBPP
- HumanEval

HumanEval only has a test split, so it can only be used for evaluation.


Except for HumanEval, all other datasets have a train, validation, and test split.
For every sample (including HumanEval), we have the following format:
```
{
    "input_text": str,
    "answer": str
}
```