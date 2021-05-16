export GLUE_DIR=/mnt/d/training_data/glue_data
export TASK_NAME=RTE

python ./examples/run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_eval \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /mnt/c/Users/kaise/Desktop/Results/evals/


python ./examples/run_glue.py \
    --model_name_or_path /mnt/d/glue_results/$TASK_NAME \
    --task_name $TASK_NAME \
    --do_eval \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /mnt/d/glue_results/$TASK_NAME \
    --attention_type standard \
    --dataset_name $TASK_NAME \
    --plot_fig 8

python ./examples/run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --dp_eval \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8 \
    --per_gpu_train_batch_size=8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /mnt/d/glue_results/$TASK_NAME \
    --attention_type standard \
    --dataset_name $TASK_NAME \
    --plot_fig -1