
DATASETS="boolq cb copa rte wic wsc gsm8k mbpp"
for DATASET in $DATASETS; do
    python3 src/run_eval.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --use_prompt_tuning false \
    --dataset $DATASET \
    --eval_size 200 \
    --output_file ./results/$DATASET\_baseline_eval_results.json
done