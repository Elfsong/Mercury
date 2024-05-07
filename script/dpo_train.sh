
# bigcode/starcoder2-3b
accelerate  launch --main_process_port 40001  ./src/dpo_train.py    \
    --model_name    bigcode/starcoder2-3b   \
    --beta          0.1  \
    --learning_rate 2e-4    \
    --max_prompt_length 1024    \
    --max_length    2048    \
    --warmup_steps  100     \
    --max_steps     200     \
    --per_device_train_batch_size   1   \
    --gradient_accumulation_steps   4

# deepseek-ai/deepseek-coder-1.3b-base
accelerate  launch --main_process_port 40002  ./src/dpo_train.py    \
    --model_name    deepseek-ai/deepseek-coder-1.3b-base   \
    --beta          0.1  \
    --learning_rate 2e-4    \
    --max_prompt_length 1024    \
    --max_length    2048    \
    --warmup_steps  100     \
    --max_steps     200     \
    --per_device_train_batch_size   1   \
    --gradient_accumulation_steps   4

# bigcode/starcoder2-7b
accelerate  launch --main_process_port 40003  ./src/dpo_train.py    \
    --model_name    deepseek-ai/deepseek-coder-1.3b-base   \
    --beta          0.1  \
    --learning_rate 2e-4    \
    --max_prompt_length 1024    \
    --max_length    2048    \
    --warmup_steps  100     \
    --max_steps     200     \
    --per_device_train_batch_size   1   \
    --gradient_accumulation_steps   4

# codellama/CodeLlama-7b-hf
accelerate  launch --main_process_port 40004  ./src/dpo_train.py    \
    --model_name    codellama/CodeLlama-7b-hf   \
    --beta          0.1  \
    --learning_rate 2e-4    \
    --max_prompt_length 1024    \
    --max_length    2048    \
    --warmup_steps  100     \
    --max_steps     200     \
    --per_device_train_batch_size   1   \
    --gradient_accumulation_steps   4

# deepseek-ai/deepseek-coder-6.7b-base
accelerate  launch --main_process_port 40005  ./src/dpo_train.py    \
    --model_name    deepseek-ai/deepseek-coder-6.7b-base   \
    --beta          0.1  \
    --learning_rate 2e-4    \
    --max_prompt_length 1024    \
    --max_length    2048    \
    --warmup_steps  100     \
    --max_steps     200     \
    --per_device_train_batch_size   1   \
    --gradient_accumulation_steps   4

# Qwen/CodeQwen1.5-7B
accelerate  launch --main_process_port 40006  ./src/dpo_train.py    \
    --model_name    Qwen/CodeQwen1.5-7B   \
    --load_in_4bit  False  \
    --beta          0.1  \
    --learning_rate 2e-4    \
    --max_prompt_length 1024    \
    --max_length    2048    \
    --warmup_steps  100     \
    --max_steps     200     \
    --per_device_train_batch_size   1   \
    --gradient_accumulation_steps   4

# bigcode/starcoder2-15b
accelerate  launch --main_process_port 40007  ./src/dpo_train.py    \
    --model_name    bigcode/starcoder2-15b   \
    --beta          0.1  \
    --learning_rate 2e-4    \
    --max_prompt_length 1024    \
    --max_length    2048    \
    --warmup_steps  100     \
    --max_steps     200     \
    --per_device_train_batch_size   1   \
    --gradient_accumulation_steps   4

# codellama/CodeLlama-13b-hf
accelerate  launch --main_process_port 40008  ./src/dpo_train.py    \
    --model_name    codellama/CodeLlama-13b-hf   \
    --beta          0.1  \
    --learning_rate 2e-4    \
    --max_prompt_length 1024    \
    --max_length    2048    \
    --warmup_steps  100     \
    --max_steps     200     \
    --per_device_train_batch_size   1   \
    --gradient_accumulation_steps   4

# codellama/CodeLlama-34b-hf
accelerate  launch --main_process_port 40009  ./src/dpo_train.py    \
    --model_name    codellama/CodeLlama-34b-hf   \
    --beta          0.1  \
    --learning_rate 2e-4    \
    --max_prompt_length 1024    \
    --max_length    2048    \
    --warmup_steps  100     \
    --max_steps     200     \
    --per_device_train_batch_size   1   \
    --gradient_accumulation_steps   4

# deepseek-ai/deepseek-coder-33b-base
accelerate  launch --main_process_port 40010  ./src/dpo_train.py    \
    --model_name    deepseek-ai/deepseek-coder-33b-base   \
    --beta          0.1  \
    --learning_rate 2e-4    \
    --max_prompt_length 1024    \
    --max_length    2048    \
    --warmup_steps  100     \
    --max_steps     200     \
    --per_device_train_batch_size   1   \
    --gradient_accumulation_steps   4