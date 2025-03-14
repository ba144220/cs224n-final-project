
DATASETS="boolq mbpp gsm8k"

for DATASET in $DATASETS; do

    EXP_LIST="nl nl_16 nl_32 nl_64 rand_16 rand_32 rand_64"
    # If dataset is boolq, only add nl_8 and rand_8 to the list
    if [ $DATASET == "boolq" ]; then
        EXP_LIST="nl_8 rand_8 $EXP_LIST"
    fi

    BATCH_SIZE=16
    if [ $DATASET == "boolq" ]; then
        BATCH_SIZE=4
    fi

    for EXP in $EXP_LIST; do
        echo "--------------------------------"
        echo "Evaluating $DATASET $EXP"
        echo "--------------------------------"
        python3 src/run_eval.py \
        --model_name meta-llama/Llama-3.2-1B-Instruct \
        --use_prompt_tuning true \
        --soft_prompt_path ./outputs/$DATASET\_$EXP/soft_prompt.pt \
        --dataset $DATASET \
        --eval_size 400 \
        --batch_size $BATCH_SIZE \
        --output_file ./results/$DATASET/$DATASET\_$EXP\_eval_results.json
    done
done

