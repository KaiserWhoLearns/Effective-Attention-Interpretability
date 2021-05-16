#!/bin/bash
# Before running this script, please make sure you ran pip install . in code_dir
read -p "Enter the training data data location(must be a path): " glue_dir
read -p "Enter the code dir(must be a path): " code_dir

# Train BERT with two different random seed
dataset_names=("MNLI" "QQP" "SNLI" "WNLI" "RTE" "MRPC" "QNLI" "SST-2" "STS-B")

for i in {1..5}
do
    random_seeds[i]=$(python generate_seed.py)
done
echo "The random seeds are: " ${random_seeds[@]}

for dataset in "${dataset_names[@]}"
do
    for seed in "${random_seeds[i]}"
    do
        python $code_dir/examples/run_glue.py \
        --model_name_or_path bert-base-uncased \
        --task_name $dataset \
        --do_train \
        --do_eval \
        --data_dir $glue_dir/$dataset \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8 \
        --per_gpu_train_batch_size=8 \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --output_dir /mnt/d/glue_results/$TASK_NAME \
        --attention_type standard \
        --dataset_name $dataset \
        --plot_fig 5 \
        --seed $seed
    done
done