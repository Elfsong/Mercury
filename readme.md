# Mercury 🪐

Welcome to Mercury: An Efficiency Benchmark for LLM Code Synthesis

* Mercury is the first code efficiency benchmark designed for code synthesis tasks.

* It consists of 1,889 programming tasks covering diverse difficulty levels, along with test case generators that produce unlimited cases for comprehensive evaluation. 

You can find our paper at: https://arxiv.org/abs/2402.07844

The dataset is available at: https://huggingface.co/datasets/Elfsong/Mercury

## How to use Mercury
```python
# set up OpenAI key if you are going to evaluate it
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
