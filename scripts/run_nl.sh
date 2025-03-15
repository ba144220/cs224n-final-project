
DATASET=$1
SIZES="16 32 64"


python3 src/run_finetune.py \
--model_name "meta-llama/Llama-3.2-1B-Instruct" \
--use_prompt_tuning true \
--init_from_natural_language true \
--dataset $DATASET \
--train_size 2000 \
--eval_size 100 \
--output_dir ./outputs/$DATASET\_nl \
--do_train true \
--do_eval true \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 2 \
--learning_rate 2.0e-5 \
--weight_decay 0.01 \
--warmup_steps 50 \
--logging_steps 10 \
--eval_strategy "steps" \
--eval_steps 100 \
--save_strategy "steps" \
--save_steps 200 \
--save_total_limit 1

for SIZE in $SIZES; do
python3 src/run_finetune.py \
--model_name "meta-llama/Llama-3.2-1B-Instruct" \
--use_prompt_tuning true \
--init_from_natural_language true \
--prompt_tuning_length $SIZE \
--dataset $DATASET \
--train_size 2000 \
--eval_size 100 \
--output_dir ./outputs/$DATASET\_nl_${SIZE} \
--do_train true \
--do_eval true \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 2 \
--learning_rate 2.0e-5 \
--weight_decay 0.01 \
--warmup_steps 50 \
--logging_steps 10 \
--eval_strategy "steps" \
--eval_steps 100 \
--save_strategy "steps" \
--save_steps 200 \
--save_total_limit 1
done
