# Fine-Tune Llama2-7b on SE paired dataset

# 0. imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

import torch
import datasets
from tqdm import tqdm
from typing import Optional
from accelerate import Accelerator
from accelerate import PartialState
from dataclasses import dataclass, field
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer
from trl.import_utils import is_xpu_available

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="bigcode/starcoder2-7b", metadata={"help": "the model name"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "Model 4 bit quant"})
    
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})

    dataset_name: Optional[str] = field(default="Elfsong/Mercury", metadata={"help": "the dataset name"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=300, metadata={"help": "the size of the validation set"})

    seq_length: Optional[int] = field(default=4096, metadata={"help": "the sequence length"})
    max_steps: Optional[int] = field(default=800, metadata={"help": "the maximum number of sgd steps"})
    logging_steps: Optional[int] = field(default=4, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=1000, metadata={"help": "the saving frequency"})
    
    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=4, metadata={"help": "the gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "whether to use gradient checkpointing"})
    
    group_by_length: Optional[bool] = field(default=False, metadata={"help": "whether to group by length"})
    packing: Optional[bool] = field(default=False, metadata={"help": "whether to use packing for SFTTrainer"})

    lora_alpha: Optional[float] = field(default=8, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    num_warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    output_dir: Optional[str] = field(default="/home/mingzhe/Projects/Mercury/checkpoints", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if script_args.group_by_length and script_args.packing:
    raise ValueError("Cannot use both packing and group by length")

print("Model Loading...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=script_args.load_in_4bit,
    load_in_8bit=not script_args.load_in_4bit, 
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    # device_map={"": PartialState().process_index},
    device_map="auto",
    trust_remote_code=True,
    token=True,
)
base_model.config.use_cache = False
print("Model Loaded.")

peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_steps=script_args.num_warmup_steps,
    optim=script_args.optimizer_type,
    bf16=True,
    remove_unused_columns=True,
    run_name=f"sft_train_{script_args.model_name}",
    gradient_checkpointing=script_args.gradient_checkpointing,
)

print("Downloading dataset...")
dataset = datasets.load_dataset(script_args.dataset_name)

def prompt_generate(question_content, starter_code="", answer=""):
    examples_json = {
        "question": "You are given a 0-indexed array of positive integers nums. Find the number of triplets (i, j, k) that meet the following conditions:\n\n0 <= i < j < k < nums.length\nnums[i], nums[j], and nums[k] are pairwise distinct.\n\t\nIn other words, nums[i] != nums[j], nums[i] != nums[k], and nums[j] != nums[k].\n\n\n\nReturn the number of triplets that meet the conditions.\n \nExample 1:\n\nInput: nums = [4,4,2,4,3]\nOutput: 3\nExplanation: The following triplets meet the conditions:\n- (0, 2, 4) because 4 != 2 != 3\n- (1, 2, 4) because 4 != 2 != 3\n- (2, 3, 4) because 2 != 4 != 3\nSince there are 3 triplets, we return 3.\nNote that (2, 0, 4) is not a valid triplet because 2 > 0.\n\nExample 2:\n\nInput: nums = [1,1,1,1,1]\nOutput: 0\nExplanation: No triplets meet the conditions so we return 0.\n\n \nConstraints:\n\n3 <= nums.length <= 100\n1 <= nums[i] <= 1000\n\n",
        "sample_code": 'class Solution(object):\n    def unequalTriplets(self, nums: List[int]) -> int:\n        """\n\t:type nums: List[int]\n\t:rtype: int\n\t"""\n        \n',
        "answer": 'class Solution(object):\n    def unequalTriplets(self, nums: List[int]) -> int:\n        """\n\t:type nums: List[int]\n\t:rtype: int\n\t"""\n        \n        ans = 0\n        n = len(a)\n        for i in range(n):\n            for j in range(i + 1, n):\n                for k in range(j + 1, n):\n                    ans += len({a[i], a[j], a[k]}) == 3\n        return ans'
    }

    def get_example_prompt(example):
        prompt = ""
        prompt += "### Question\n"
        prompt += example["question"]
        prompt += "\n\n"
        if starter_code:
            prompt += "### Code Prompt\n"
            prompt += example["sample_code"]
            prompt += "\n\n"
        prompt += "### Completion\n"
        prompt += example["answer"]
        if example["answer"]:
            prompt += "\n\n"
        return prompt

    prompt = ""
    # one-shot generation example
    # prompt += get_example_prompt(examples_json)
    # code generation
    prompt += get_example_prompt({"question": question_content,"sample_code": starter_code,"answer": answer})
    
    return prompt

def formatting_prompts_func(examples):
    output_texts = []
    for i in range(len(examples["pretty_content"])):
        if len(examples["pretty_content"][i]) > 0:
            content = examples["pretty_content"][i][0]
            starter = examples["prompt"][i]
            for s in examples["solutions"][i]:
                solution = s['solution']
                prompt = prompt_generate(content, starter, solution)
                output_texts.append(prompt)
                # output_texts.append(f'{content}\n{solution}')
    return output_texts


trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['eval'],
    peft_config=peft_config,
    max_seq_length=script_args.seq_length,
    formatting_func=formatting_prompts_func,
    tokenizer=tokenizer,
    args=training_args,
)
trainer.train()

output_dir = os.path.join(script_args.output_dir, f"{script_args.model_name}-sft-final_checkpoint")
trainer.save_model(output_dir)

model = trainer.model.merge_and_unload()
model.save_pretrained(output_dir)

# Free memory for merging weights
del base_model
if is_xpu_available():
    torch.xpu.empty_cache()
else:
    torch.cuda.empty_cache()

# [Warning] DeepSpeed Zero-3 is not compatible with `low_cpu_mem_usage=True` or with passing a `device_map`.
# Do this operation later manually
# model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
# model = model.merge_and_unload()

# output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
# model.save_pretrained(output_merged_dir, safe_serialization=True)