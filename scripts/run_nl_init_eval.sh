
DATASET=$1
python3 src/run_eval.py \
--model_name meta-llama/Llama-3.2-1B-Instruct \
--soft_prompt_path ./output/boolq_prompt_tuning_dry_run/checkpoint-100/soft_prompt.pt \
--use_prompt_tuning true \
--dataset $DATASET \
--eval_size 10 \
--output_file ./results/$DATASET\_nl_init_eval_results.json