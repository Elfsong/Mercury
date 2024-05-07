# Author: Mingzhe Du (mingzhe@nus.edu.sg)
# Date: 20 / 04 / 2024

# bigcode/starcoder2-3b
python ./src/sft_train.py \
    --model_name    bigcode/starcoder2-3b    \
    --seq_length    2048   \
    --max_steps     200     \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4

# deepseek-ai/deepseek-coder-1.3b-base
python ./src/sft_train.py \
    --model_name    deepseek-ai/deepseek-coder-1.3b-base    \
    --seq_length    2048   \
    --max_steps     200     \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4

# bigcode/starcoder2-7b
python ./src/sft_train.py \
    --model_name    bigcode/starcoder2-7b    \
    --seq_length    2048   \
    --max_steps     200     \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4

# codellama/CodeLlama-7b-hf
python ./src/sft_train.py \
    --model_name    codellama/CodeLlama-7b-hf    \
    --seq_length    2048   \
    --max_steps     200     \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4

# deepseek-ai/deepseek-coder-6.7b-base
python ./src/sft_train.py \
    --model_name    deepseek-ai/deepseek-coder-6.7b-base    \
    --seq_length    2048   \
    --max_steps     200     \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4

# Qwen/CodeQwen1.5-7B
python ./src/sft_train.py \
    --model_name    Qwen/CodeQwen1.5-7B    \
    --load_in_4bit False    \
    --seq_length    2048   \
    --max_steps     200     \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4

# bigcode/starcoder2-15b
python ./src/sft_train.py \
    --model_name    bigcode/starcoder2-15b    \
    --seq_length    2048   \
    --max_steps     200     \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4

# codellama/CodeLlama-13b-hf
python ./src/sft_train.py \
    --model_name    codellama/CodeLlama-13b-hf    \
    --seq_length    2048   \
    --max_steps     100     \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4


# codellama/CodeLlama-34b-hf
python ./src/sft_train.py \
    --model_name    codellama/CodeLlama-34b-hf   \
    --seq_length    2048   \
    --max_steps     200     \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4

# deepseek-ai/deepseek-coder-33b-base
python ./src/sft_train.py \
    --model_name    deepseek-ai/deepseek-coder-33b-base   \
    --seq_length    2048   \
    --max_steps     200     \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4