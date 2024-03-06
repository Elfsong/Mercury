# Mercury 🪐

Welcome to Mercury: An Efficiency Benchmark for LLM Code Synthesis

* Mercury is the first code efficiency benchmark designed for code synthesis tasks.

* It consists of 1,889 programming tasks covering diverse difficulty levels, along with test case generators that produce unlimited cases for comprehensive evaluation. 

You can find our paper at: https://arxiv.org/abs/2402.07844

The dataset is available at: https://huggingface.co/datasets/Elfsong/Mercury

## How to use Mercury
```python
# set up OpenAI key if you are going to evaluate it (optional)
import os
os.environ["OPENAI_API_KEY"] = 'YOUR_OPENAI_KEY'

# Instantiate evaluator with model_name
# Set do_generate to True if you are going to load the specific language model during evaluator initialization.
from src import evaluator as Evaluator
evaluator = Evaluator.DistributeWiseEvaluator(model_name_or_path='openai/gpt-3.5-turbo-1106', do_generate=True)

# Generate code samples
evaluator.generate(num_samples_per_task=1)

# Evaluate code samples using the Mercury benchmark
evaluator.evaluate(num_samples_per_task=1)
```

# Leaderboard (WIP 🚧)

| Model Name           | Pass@1 | Beyond@1 |
| -------------------- | ------ | -------- |
| gpt-3.5-turbo-1106   | 0.8711 | 0.7214   |
| gpt-4-1106-preview   | 0.7930 | 0.6555   |
| deepseek-coder-6.7b  | 0.6875 | 0.5492   |
| deepseek-coder-33b   | 0.7734 | 0.6951   |
| CodeLlama-7b         | 0.3672 | 0.2798   |
| CodeLlama-13b        | 0.4023 | 0.3091   |
| CodeLlama-34b        | 0.5703 | 0.4505   |
