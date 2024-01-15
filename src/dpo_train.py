# coding: utf-8

import os
os.environ['TRANSFORMERS_CACHE'] = '/mnt/dataDisk1/huggingface_cache'

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
    model_name_or_path: Optional[str] = field(default="deepseek-ai/deepseek-coder-1.3b-instruct", metadata={"help": "the location of the SFT model name or path"})
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=96, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(default=4, metadata={"help": "the number of gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "whether to use gradient checkpointing"})

    lora_alpha: Optional[float] = field(default=8, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=600, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1600, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=600, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=8, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=200, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=10000, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="/mnt/dataDisk1/checkpoints/dpo", metadata={"help": "the output directory"})
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

def get_code_paired(split="train", sanity_check: bool = False):
    dataset = load_dataset("Elfsong/leetcode_v4", split=split)
    
    starter = "class Solution"
    
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))
    
    data = list()

    for question in dataset:
        content = question['pretty_content']
        solutions = question['solutions']
        code_prompt = question['prompt']
        
        for pair in list(permutations(solutions, 2) ):
            a, b = pair
            a_time = int(a["runtime"][:-2])
            b_time = int(b["runtime"][:-2])

            if b_time - a_time > 15:
                chosen_code, rejected_code = a["solution"], b["solution"]
                
                if (starter in chosen_code) and (starter in rejected_code):
                    prompt = code_prompt
                    content = content
                    
                    data += [{
                        "prompt": f"[INST] Complete python code to solve the following coding problem:\n {content} \n {prompt} \n [/INST] \n ",
                        "chosen": f"[CODE] {chosen_code} [/CODE]",
                        "rejected": f"[CODE] {rejected_code} [/CODE]",
                    }]
    
    return Dataset.from_list(data)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map={"": Accelerator().local_process_index},
        trust_remote_code=True,
        token=True,
        low_cpu_mem_usage=True,
        cache_dir="/mnt/dataDisk1/huggingface_cache"
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool]
    
    
    tokenizer = AutoTokenizer.from_pretrained("Phind/Phind-CodeLlama-34B-v2", legacy=False)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the Stack-exchange paired dataset    
    train_dataset = get_code_paired(split="train", sanity_check=script_args.sanity_check)
    print(f"Before: {len(train_dataset)}")
    train_dataset = train_dataset.filter(lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length)
    print(f"After: {len(train_dataset)}")

    # 3. Load evaluation dataset
    eval_dataset = get_code_paired(split="eval", sanity_check=script_args.sanity_check)
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    )

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
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)