# coding: utf-8

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
from trl import DPOTrainer
from peft import LoraConfig
from typing import Dict, Optional
from accelerate import Accelerator
from itertools import permutations
from dataclasses import dataclass, field
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, BitsAndBytesConfig


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(default="deepseek-ai/deepseek-coder-33b-instruct", metadata={"help": "the location of the SFT model name or path"})
    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=96, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "Model 4 bit quant"})

    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(default=4, metadata={"help": "the number of gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "whether to use gradient checkpointing"})

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=768, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1500, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=800, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=2, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=1000, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=1000, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="/home/mingzhe/Projects/Mercury/checkpoints", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on a few samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    
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
    prompt += get_example_prompt(examples_json)
    # code generation
    prompt += get_example_prompt({"question": question_content,"sample_code": starter_code,"answer": answer})
    
    return prompt

def get_code_paired(split="train", sanity_check: bool = False):
    dataset = load_dataset("Elfsong/Mercury", split=split)
    
    starter = "class Solution"
    
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))
    
    data = list()

    for question in dataset:
        if question['pretty_content']:
            content = question['pretty_content'][0]
            solutions = question['solutions']
            code_prompt = question['prompt']
            
            for pair in list(permutations(solutions, 2) ):
                a, b = pair
                a_time = int(a["runtime"][:-2])
                b_time = int(b["runtime"][:-2])

                if b_time - a_time > 20:
                    chosen_code, rejected_code = a["solution"], b["solution"]
                    
                    if (starter in chosen_code) and (starter in rejected_code):                    
                        data += [{
                            "prompt": prompt_generate(content, code_prompt),
                            "chosen": f"{chosen_code}",
                            "rejected": f"{rejected_code}",
                        }]
    
    return Dataset.from_list(data)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=script_args.load_in_4bit,
        load_in_8bit=not script_args.load_in_4bit, 
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=bnb_config,
        # device_map={"": Accelerator().local_process_index},
        device_map="auto",
        trust_remote_code=True,
        token=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool]
    
    
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, legacy=False)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load training dataset    
    train_dataset = get_code_paired(split="train", sanity_check=script_args.sanity_check)
    print(f"Train Dataset Before: {len(train_dataset)}")
    train_dataset = train_dataset.filter(lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length)
    print(f"Train Dataset After: {len(train_dataset)}")

    # # 3. Load evaluation dataset
    # eval_dataset = get_code_paired(split="eval", sanity_check=script_args.sanity_check)
    # print(f"Eval Dataset Before: {len(eval_dataset)}")
    # eval_dataset = eval_dataset.filter(lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length)
    # print(f"Eval Dataset After: {len(eval_dataset)}")

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        remove_unused_columns=False,
        bf16=True,
        run_name=f"dpo_train_{script_args.model_name_or_path}",
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        # model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=train_dataset, # for faster loading, cause we don't evaluate here.
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

    # 6. train
    dpo_trainer.train()
    
    # 7. save
    output_dir = os.path.join(script_args.output_dir, f"{script_args.model_name_or_path}-dpo-final_checkpoint")
    dpo_trainer.save_model(output_dir)

    model = dpo_trainer.model.merge_and_unload()
    model.save_pretrained(output_dir)