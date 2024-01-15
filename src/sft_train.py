# Fine-Tune Llama2-7b on SE paired dataset

# 0. imports
import os
os.environ['TRANSFORMERS_CACHE'] = '/mnt/dataDisk1/huggingface_cache'

from dataclasses import dataclass, field
from typing import Optional

import torch
import datasets
from accelerate import Accelerator
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer
from trl.import_utils import is_xpu_available

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="deepseek-ai/deepseek-coder-1.3b-instruct", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})

    dataset_name: Optional[str] = field(default="Elfsong/leetcode_v4", metadata={"help": "the dataset name"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=300, metadata={"help": "the size of the validation set"})

    seq_length: Optional[int] = field(default=8192, metadata={"help": "the sequence length"})
    max_steps: Optional[int] = field(default=1000, metadata={"help": "the maximum number of sgd steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=500, metadata={"help": "the saving frequency"})
    
    per_device_train_batch_size: Optional[int] = field(default=6, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=4, metadata={"help": "the gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "whether to use gradient checkpointing"})
    
    group_by_length: Optional[bool] = field(default=False, metadata={"help": "whether to group by length"})
    packing: Optional[bool] = field(default=False, metadata={"help": "whether to use packing for SFTTrainer"})

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    num_warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    output_dir: Optional[str] = field(default="/mnt/dataDisk1/checkpoints/sft", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if script_args.group_by_length and script_args.packing:
    raise ValueError("Cannot use both packing and group by length")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def dataset_convert(tokenizer, dataset):
    sft_dataset = list()
    for instance in dataset:
        content = instance["pretty_content"]
        prompt =  instance["prompt"]
        
        for solution in instance["solutions"]:
            code = solution["solution"]
            if "class Solution" in code:
                prompt_template = f"<INS>\n{content}\n{prompt}\n<\\INS>\n<CODE>\n{code}\n<\\CODE>"                
                sft_dataset += [{
                    "input_ids":tokenizer.encode(prompt_template, add_special_tokens=True, return_tensors="pt"),
                    "text": prompt_template,
                }]
    
    return datasets.Dataset.from_list(sft_dataset)

def create_datasets(tokenizer, args):
    dataset = datasets.load_dataset(args.dataset_name)
    # dataset = dataset.train_test_split(test_size=0.1, seed=None)
    
    train_data = dataset["train"]
    valid_data = dataset["eval"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    
    train_dataset = dataset_convert(tokenizer, train_data)
    valid_dataset = dataset_convert(tokenizer, valid_data)
    
    return train_dataset, valid_dataset
    
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    device_map={"": Accelerator().local_process_index},
    trust_remote_code=True,
    token=True,
    cache_dir="/mnt/dataDisk1/huggingface_cache"
)
base_model.config.use_cache = False

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
    remove_unused_columns=False,
    run_name="sft_train",
    gradient_checkpointing=script_args.gradient_checkpointing,
)

train_dataset, eval_dataset = create_datasets(tokenizer, script_args)
train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) <= script_args.seq_length)


trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    packing=script_args.packing,
    dataset_text_field="text",
    max_seq_length=script_args.seq_length,
    tokenizer=tokenizer,
    args=training_args,
)
trainer.train()
trainer.save_model(script_args.output_dir)

output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

# Free memory for merging weights
del base_model
if is_xpu_available():
    torch.xpu.empty_cache()
else:
    torch.cuda.empty_cache()

# DeepSpeed Zero-3 is not compatible with `low_cpu_mem_usage=True` or with passing a `device_map`.
# Do this operation later manually
# model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
# model = model.merge_and_unload()

# output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
# model.save_pretrained(output_merged_dir, safe_serialization=True)