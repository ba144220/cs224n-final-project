
DATASET=$1
python3 src/run_eval.py \
--model_name meta-llama/Llama-3.2-1B-Instruct \
--use_prompt_tuning false \
--dataset $DATASET \
--eval_size 10 \
--output_file ./results/$DATASET\_baseline_eval_results.json