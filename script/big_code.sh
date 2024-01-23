accelerate launch  main.py \
  --model codellama/CodeLlama-13b-Instruct-hf \
  --max_length_generation 512 \
  --tasks mbpp \
  --temperature 0.2 \
  --n_samples 1 \
  --batch_size 1 \
  --load_in_4bit \
  --allow_code_execution \
  --save_generations