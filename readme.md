# Mercury: A Code Efficiency Benchmark for Code Large Language Models 🪐

* Welcome to Mercury!
* Mercury is the first code efficiency benchmark designed for code synthesis tasks.
* It consists of 1,889 programming tasks covering diverse difficulty levels, along with test case generators that produce unlimited cases for comprehensive evaluation.

> [July, 10, 2024] We are building [**CodeArena**](https://codellm.club/about/) now for more efficient Code LLMs evaluation!

> [June 24, 2024] We are currently working on the [**Multilingual Mercury**](https://huggingface.co/datasets/Elfsong/Mercury_Multilingual) 🚧

> [May 26, 2024] Mercury is now available on [**BigCode**](https://github.com/bigcode-project/bigcode-evaluation-harness/tree/main/docs#mercury) 🌟

## Mercury Datasets Access
We publish and maintain our datasets at [**Mercury@HF**](https://huggingface.co/datasets/Elfsong/Mercury)

## How to use Mercury Evaluation
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

## Citation
```
@article{du2024mercury,
  title={Mercury: An Efficiency Benchmark for LLM Code Synthesis},
  author={Du, Mingzhe and Luu, Anh Tuan and Ji, Bin and Ng, See-Kiong},
  journal={arXiv preprint arXiv:2402.07844},
  year={2024}
}
```

## Questions?
Should you have any questions regarding this paper, please feel free to email us (mingzhe@nus.edu.sg). Thank you for your attention!

