# Mercury 🪐

Welcome to Mercury: An Efficiency Benchmark for LLM Code Synthesis

* Mercury is the first code efficiency benchmark designed for code synthesis tasks.

* It consists of 1,889 programming tasks covering diverse difficulty levels, along with test case generators that produce unlimited cases for comprehensive evaluation. 

## Important Update
To ensure the preservation of so-called 'anonymity', the dataset link will no longer be available on this page prior to the formal paper acceptance. Sorry for any inconvenience this may cause. Researchers are encouraged to find the dataset independently.

I learned a big lesson that the meticulous nature of some reviewers is akin to that of 'detectives', taking pride in their ability to unearth authorship through a series of multi-hop reasoning:)

## How to use Mercury
```shell
# Option 1 (with BigCode):
# See https://github.com/bigcode-project/bigcode-evaluation-harness/tree/main/docs#mercury
accelerate  launch --main_process_port 30003  main.py  \
    --model bigcode/starcoder2-7b   \
    --load_in_4bit   \
    --max_length_generation 2048   \
    --tasks mercury    \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 5   \
    --allow_code_execution  \
    --save_generations  \
    --metric_output_path starcoder2-7b-mercury-result.json
```

```python
# Option 2 (this library):
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
## Benchmark Visualization
![mercury_benchmark](https://github.com/Elfsong/Mercury/assets/12135272/4b4b2126-a06c-43dc-ae16-2848d9f77a69)



