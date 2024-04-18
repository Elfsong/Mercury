accelerate launch  main.py \
  --model codellama/CodeLlama-7b-Instruct-hf \
  --limit 10 \
  --max_length_generation 1024 \
  --tasks mercury \
  --n_samples 1 \
  --batch_size 1 \
  --allow_code_execution \
  --save_generations