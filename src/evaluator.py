# coding: utf-8

import os
import json
import gzip
import torch
import argparse
import itertools
import numpy as np

from tqdm import tqdm
from sandbox import Sandbox
from datasets import load_dataset
from typing import Iterable, Dict, List, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from human_eval.data import read_problems, write_jsonl, stream_jsonl

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False

class Evaluator(object):
    def __init__(self, model_name_or_path) -> None:
        assert model_name_or_path is not None
        self.model_name_or_path = model_name_or_path
    
    @staticmethod
    def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
        """
        Writes an iterable of dictionaries to jsonl
        """
        if append:
            mode = 'ab'
        else:
            mode = 'wb'
        filename = os.path.expanduser(filename)
        if filename.endswith(".gz"):
            with open(filename, mode) as fp:
                with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                    for x in data:
                        gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
        else:
            with open(filename, mode) as fp:
                for x in data:
                    fp.write((json.dumps(x) + "\n").encode('utf-8'))
                
    @staticmethod
    def estimate_pass_at_k(num_samples: Union[int, List[int], np.ndarray], num_correct: Union[List[int], np.ndarray], k: int):
        """ Estimates pass@k of each problem and returns them in an array. """

        def estimator(n: int, c: int, k: int) -> float:
            """
            Calculates 1 - comb(n - c, k) / comb(n, k).
            """
            if n - c < k:
                return 1.0
            return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

        if isinstance(num_samples, int):
            num_samples_it = itertools.repeat(num_samples, len(num_correct))
        else:
            assert len(num_samples) == len(num_correct)
            num_samples_it = iter(num_samples)

        return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])
    
    def generate_completion(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        generate_ids = self.model.generate(
            inputs.input_ids.to("cuda"), 
            attention_mask=inputs.attention_mask.to("cuda"), 
            pad_token_id=self.tokenizer.eos_token_id, 
            max_new_tokens=512, 
            do_sample=True, 
            top_p=0.75, 
            top_k=40, 
            temperature=0.1
        )
        completion = self.tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        completion = completion.replace(prompt, "").split("\n\n\n")[0]

        return completion

    def generate_completions(self, prompt: str, n_sample=1):
        inputs = self.tokenizer([prompt] * n_sample, return_tensors="pt", truncation=True, max_length=4096)
        generate_ids = self.model.generate(
            inputs.input_ids.to("cuda"), 
            attention_mask=inputs.attention_mask.to("cuda"), 
            pad_token_id=self.tokenizer.eos_token_id, 
            max_new_tokens=512, 
            do_sample=True, 
            top_p=0.75, 
            top_k=40, 
            temperature=0.1,
        )
        completions = self.tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False)
        completions = list(map(lambda x: x.replace(prompt, "").split("\n\n\n")[0], completions))

        return completions

    def generate(self):
        raise NotImplementedError("Don't call the interface directly")
    
    def evaluate(self):
        raise NotImplementedError("Don't call the interface directly")
    
class HumanEvalEvaluator(Evaluator):
    def __init__(self, model_name_or_path, sample_file=None) -> None:
        super().__init__(model_name_or_path)
        self.problems = read_problems()
        self.sample_file = sample_file
        self.save_name = self.model_name_or_path.split("/")[-1]
        self.num_samples_per_task = 10
        
    def generate(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, device_map="auto", cache_dir="/mnt/dataDisk1/huggingface_cache")         
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        samples = list()
        
        # for task_id in tqdm(list(self.problems.keys())[:10]):
        for task_id in tqdm(self.problems):
            for completion in self.generate_completions(self.problems[task_id]["prompt"], self.num_samples_per_task):
                samples += [{
                    "task_id": task_id,
                    "completion": completion,
                }]
        
        write_jsonl(f"{self.save_name}_samples.jsonl", samples)
        return samples
    
    def evaluate(self, samples=None):        
        if not samples:
            assert self.sample_file
            samples = list()
            for sample in tqdm(stream_jsonl(self.sample_file)):
                sample["problem"] = self.problems[sample["task_id"]]
                samples += [sample]
            
        results, n_samples = Sandbox.run_samples(samples, n_workers=4, timeout=10.0)
        
        # Calculate pass@k.
        total, correct = [], []
        for result in results.values():
            result.sort()
            passed = [r[1]["passed"] for r in result]
            total.append(len(passed))
            correct.append(sum(passed))
        total = np.array(total)
        correct = np.array(correct)

        ks = [1,10]
        pass_at_k = {f"pass@{k}": HumanEvalEvaluator.estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}
        
        return pass_at_k
    
    
class OnlineEvaluator(Evaluator):
    def __init__(self, model_name_or_path) -> None:
        super().__init__(model_name_or_path)
        self.dataset = load_dataset("Elfsong/leetcode_v4", split='train[:3]')
        self.save_name = self.model_name_or_path.split("/")[-1]
        self.num_samples_per_task = 10
    
    @staticmethod
    def sample_creator(instance):
        try:
            prompt = f"<INS>\n{instance['pretty_content'][0]}\n{instance['prompt']}<\INS>"
            return prompt
        except:
            pass
        
    def generate(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, device_map="auto", cache_dir="/mnt/dataDisk1/huggingface_cache")         
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        samples = list()
                
        for instance in tqdm(self.dataset):
            prompt = OnlineEvaluator.sample_creator(instance)
            for completion in self.generate_completions(prompt, self.num_samples_per_task):
                samples += [{
                    "task_id": instance['slug_name'],
                    "completion": completion,
                }]
            
                
        write_jsonl(f"{self.save_name}_samples.jsonl", samples)
        return samples
        
    def evaluate(self, samples=None):        
        if not samples:
            assert self.sample_file
            samples = list()
            for sample in tqdm(stream_jsonl(self.sample_file)):
                sample["problem"] = self.problems[sample["task_id"]]
                samples += [sample]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ultramarine Evaluation Framework')
    parser.add_argument('--model_name_or_path', default="deepseek-ai/deepseek-coder-1.3b-instruct", help="model name or path (huggingface or checkpoints)")
    parser.add_argument('--benchmark', default='UltraMaine', help="evaluation benchmarks")
    parser.add_argument('--samples', default=None, help="generation samples")
    parser.add_argument('--do_generate', action='store_true', help="run generation")
    parser.add_argument('--do_evaluate', action='store_true', help="run evaluation")
    
    args = parser.parse_args()
        
    print(f"Current model: [{args.model_name_or_path}] Current benchmark: [{args.benchmark}]")
    if args.benchmark == "HumanEval":
        evaluator = HumanEvalEvaluator(args.model_name_or_path, args.samples)
    elif args.benchmark == "UltraMaine":
        evaluator = OnlineEvaluator(args.model_name_or_path)
        
    if args.do_generate:
        output = evaluator.generate()
    if args.do_evaluate:
        output = evaluator.evaluate()
        print(output)
    
    print("Bingo!")