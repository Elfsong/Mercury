accelerate  launch --main_process_port 29504  ./src/sft_train.py \
    --model_name deepseek-ai/deepseek-coder-1.3b-base   \
    --seq_length 2048   \
    --max_steps 300     \
    --save_steps 100    \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-4
