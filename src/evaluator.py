# coding: utf-8

# Strik on the assigned GPU.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = 'false'
os.environ['HF_HOME'] = '/home/mingzhe/hf_cache'


import json
import gzip
import torch
import random
import argparse
import itertools
import numpy as np

from tqdm import tqdm
from openai import OpenAI
from src.sandbox import Sandbox
from datasets import load_dataset
from collections import defaultdict
from typing import Iterable, Dict, List, Union
from human_eval.data import read_problems, write_jsonl, stream_jsonl
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class Evaluator(object):
    def __init__(self, model_name_or_path) -> None:
        assert model_name_or_path is not None
        self.model_name_or_path = model_name_or_path
        self.openai_key = os.environ.get('OPENAI_API_KEY')
        if self.openai_key:
            self.openai_client = OpenAI(api_key=self.openai_key)
        else:
            raise Exception('Unknown OpenAI Key')
    
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
    
    def prompt_generate(self, instance):
        content = instance['pretty_content'][0]
        code_prompt = instance['prompt']
        if self.model_name_or_path in ['deepseek-ai/deepseek-coder-33b-instruct', 'deepseek-ai/deepseek-coder-6.7b-instruct']:
            prompt = f"Complete python3 code to solve the following coding problem:\n{content}\n{code_prompt}"
            return prompt
        elif self.model_name_or_path in ['codellama/CodeLlama-34b-Instruct-hf', 'codellama/CodeLlama-13b-Instruct-hf', 'codellama/CodeLlama-7b-Instruct-hf']:
            prompt = f"Complete python3 code to solve the following coding problem.\n{content}\n{code_prompt}"
            return prompt
        elif self.model_name_or_path in ['openai/gpt-4-1106-preview', 'openai/gpt-3.5-turbo-1106']:
            prompt = f"Complete python3 code to solve the following coding problem in the JSON format:{{'completion': '<completion_starts_with_prompt>'}}\nProblem:\n{content}\nPrompt:\n{code_prompt}"
            return prompt
        elif self.model_name_or_path in ['bigcode/starcoder2-3b']:
            prompt = f"{content}\n{code_prompt}"
            return prompt
        else:
            raise NameError(f"Can't find prompt template for [{self.model_name_or_path}]")
        
    def generate_completion(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        generate_ids = self.model.generate(
            inputs.input_ids.to("cuda"), 
            attention_mask=inputs.attention_mask.to("cuda"), 
            pad_token_id=self.tokenizer.eos_token_id, 
            max_new_tokens=1024, 
            do_sample=True, 
            top_p=0.75, 
            top_k=40, 
            temperature=0.1
        )
        completion = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        completion = completion.replace(prompt, "").split("\n\n\n")[0]

        return completion

    def generate_completions(self, prompt: str, n_sample=1):
        inputs = self.tokenizer([prompt] * n_sample, return_tensors="pt", truncation=True, max_length=4096)
        generate_ids = self.model.generate(
            inputs.input_ids.to("cuda"), 
            attention_mask=inputs.attention_mask.to("cuda"), 
            pad_token_id=self.tokenizer.eos_token_id, 
            max_new_tokens=1024, 
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

    def generate_completion_openai(self, prompt: str):
        response = self.openai_client.chat.completions.create(
            model=self.model_name_or_path.split('/')[-1],
            response_format={ "type": "json_object" },
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a top-tier leetcode expert designed to output JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
        
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
        self.dataset = load_dataset("Elfsong/leetcode_v4")
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
                
class PairWiseEvaluator(Evaluator):
    def __init__(self, model_name_or_path, do_generate) -> None:
        super().__init__(model_name_or_path)
        self.model_name_or_path = model_name_or_path
        self.do_generate = do_generate
        
        # Load dataset
        self.dataset = load_dataset("Elfsong/Caduceus_v8")
        
        if self.do_generate and self.model_name_or_path not in ['openai/gpt-4-1106-preview', 'openai/gpt-3.5-turbo-1106']:
            # BitsAndBytes config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            
            # Load model and tokenizer        
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                cache_dir=os.environ.get('HF_HOME'),
                quantization_config=bnb_config,
                trust_remote_code=True,
                device_map = "auto",
            )
        
        # Load sandbox
        self.sandbox = Sandbox()
            
    def generate(self, num_samples_per_task=1):
        # Code Completion
        samples = defaultdict(list)
        for instance in tqdm(self.dataset['eval'].to_list()):
            prompt = self.prompt_generate(instance)
            
            if self.model_name_or_path in ['openai/gpt-4-1106-preview', 'openai/gpt-3.5-turbo-1106']:
                # Online Model Call
                for _ in range(num_samples_per_task):
                    completion = self.generate_completion_openai(prompt)
                    try:
                        json_completion = json.loads(completion)
                        samples[instance['slug_name']] += [{
                            "task_id": instance['slug_name'],
                            "completion": json_completion['completion'],
                        }]
                    except Exception as e:
                        print(f"Error: {e}")
                        samples[instance['slug_name']] += [{
                            "task_id": instance['slug_name'],
                            "completion": completion,
                        }]
            else:
                # Local Model Call
                for completion in self.generate_completions(prompt, num_samples_per_task):
                    samples[instance['slug_name']] += [{
                        "task_id": instance['slug_name'],
                        "completion": (instance['prompt'] + completion).strip(),
                    }]
        
        # Dump samples
        with open(f'./data/{self.model_name_or_path}_samples.json', 'w') as sample_f:
            sample_f.write(json.dumps(samples))
            
        return samples
    
    def evaluate(self, k=1):
        # Load samples from dump
        with open(f'./data/{self.model_name_or_path}_samples.json', 'r') as sample_f:
            samples = json.load(sample_f)
        
        # Run in sandbox
        eval_results = defaultdict(list)
        for instance in tqdm(self.dataset['eval'].to_list()):
            slug_name = instance['slug_name']
            solutions = samples[slug_name]
            selected_solutions = random.sample(solutions,k) if k <= len(solutions) else solutions
            
            for index, solution in enumerate(selected_solutions):       
                sample = {
                    "solution": solution,
                    "convert_offline": instance['convert_offline'],
                    "evaluate_offline": instance['evaluate_offline'],
                    "entry_point": instance['entry_point'],
                    "test_cases": json.loads(instance['test_cases']),
                    "solution_index": index,
                    "timeout": 30,
                }
    
                result = self.sandbox.run_sample(sample)
                eval_results[slug_name] += [{
                    "slug_name": slug_name,
                    "result":  result,
                    "solution": solution,
                }]
        
        # Dump samples
        with open(f'./data/{self.model_name_or_path}_eval.json', 'w') as sample_f:
            sample_f.write(json.dumps(eval_results))

        return samples
    
class DistributeWiseEvaluator(Evaluator):
    def __init__(self, model_name_or_path, do_generate) -> None:
        super().__init__(model_name_or_path)
        self.model_name_or_path = model_name_or_path
        self.do_generate = do_generate
        
        # Load dataset
        self.dataset = load_dataset("Elfsong/Mercury")
        
        if self.do_generate and self.model_name_or_path not in ['openai/gpt-4-1106-preview', 'openai/gpt-3.5-turbo-1106']:
            # BitsAndBytes config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            
            # Load model and tokenizer        
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                cache_dir=os.environ.get('HF_HOME'),
                quantization_config=bnb_config,
                trust_remote_code=True,
                device_map = "auto",
            )
        
        # Load sandbox
        self.sandbox = Sandbox()
            
    def generate(self, num_samples_per_task=10):
        # Code Completion
        samples = defaultdict(list)
        for instance in tqdm(self.dataset['eval'].to_list()):
            prompt = self.prompt_generate(instance)
            
            if self.model_name_or_path in ['openai/gpt-4-1106-preview', 'openai/gpt-3.5-turbo-1106']:
                # Online Model Call
                for _ in range(num_samples_per_task):
                    completion = self.generate_completion_openai(prompt)
                    try:
                        json_completion = json.loads(completion)
                        samples[instance['slug_name']] += [{
                            "task_id": instance['slug_name'],
                            "completion": json_completion['completion'].strip(),
                        }]
                    except Exception as e:
                        print(f"Error: {e}")
                        samples[instance['slug_name']] += [{
                            "task_id": instance['slug_name'],
                            "completion": completion,
                        }]
            else:
                # Local Model Call
                for completion in self.generate_completions(prompt, num_samples_per_task):
                    samples[instance['slug_name']] += [{
                        "task_id": instance['slug_name'],
                        "completion": (instance['prompt'] + completion).strip(),
                    }]
        
        # Dump samples
        with open(f'./data/{self.model_name_or_path}_samples.json', 'w') as sample_f:
            sample_f.write(json.dumps(samples))
            
        return samples
    
    def evaluate(self, num_samples_per_task=1):
        # Load samples from dump
        with open(f'./data/{self.model_name_or_path}_samples.json', 'r') as sample_f:
            samples = json.load(sample_f)
        
        # Run in sandbox
        eval_results = defaultdict(list)
        for instance in tqdm(self.dataset['eval'].to_list()):
            slug_name = instance['slug_name']
            solutions = samples[slug_name]
            selected_solutions = random.sample(solutions, num_samples_per_task) if num_samples_per_task <= len(solutions) else solutions
            
            # Construct runtime distribution from sample solutions
            runtimes = list()
            for index, solution in enumerate(instance['solutions']):        
                sample = {
                    "solution": solution['solution'],
                    "convert_offline": instance['convert_offline'],
                    "evaluate_offline": instance['evaluate_offline'],
                    "entry_point": instance['entry_point'],
                    "test_cases": json.loads(instance['test_cases']),
                    "solution_index": index,
                    "timeout": 30
                }
                result = self.sandbox.run_sample(sample)
                if result['result'] == "passed":
                    runtimes += [result['runtime']]
            runtimes = sorted(runtimes)
            
            # Calculate Range
            min_runtime = min(runtimes)
            max_runtime = max(runtimes)
            
            # Evaluate generated solutions
            for index, solution in enumerate(selected_solutions):       
                sample = {
                    "solution": solution['completion'],
                    "convert_offline": instance['convert_offline'],
                    "evaluate_offline": instance['evaluate_offline'],
                    "entry_point": instance['entry_point'],
                    "test_cases": json.loads(instance['test_cases']),
                    "solution_index": index,
                    "timeout": 30,
                }
    
                result = self.sandbox.run_sample(sample)
                
                # Calculate Beyond
                runtime = result['runtime'] if result['result'] == "passed" else float('inf')                    
                beyond = max_runtime - runtime
                beyond = min(beyond, 1)
                beyond = max(beyond, 0)
                beyond_precent = beyond / (max_runtime - min_runtime)
                
                eval_results[slug_name] += [{
                    "slug_name": slug_name,
                    "status":  result,
                    "solution": solution['completion'],
                    "runtimes": runtimes,
                    "beyond_p": beyond_precent,
                }]
        
        # Score
        total, passed, beyond = 0, 0, 0
        for slug_name in eval_results:
            cases = eval_results[slug_name]
            total += 1
            beyond += cases[0]['beyond_p']
            if cases[0]['status']['result'] == "passed":
                passed += 1
        passed_score = passed / total
        beyond_score = beyond / total
        print(f"Pass@1: {passed_score} Beyond@1: {beyond_score}")


        return samples
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ultramarine Evaluation Framework')
    parser.add_argument('--model_name_or_path', default="openai/gpt-4-1106-preview", help="model name or path (huggingface or checkpoints)")
    parser.add_argument('--benchmark', default='DistributeWiseMercury', help="evaluation benchmarks")
    parser.add_argument('--samples', type=int, default=1, help="generation samples")
    parser.add_argument('--do_generate', action='store_true', help="run generation")
    parser.add_argument('--do_evaluate', action='store_true', help="run evaluation")
    
    args = parser.parse_args()
        
    print(f"Current model: [{args.model_name_or_path}] Current benchmark: [{args.benchmark}]")
    if args.benchmark == "HumanEval":
        evaluator = HumanEvalEvaluator(args.model_name_or_path, args.samples)
    elif args.benchmark == "UltraMaine":
        evaluator = OnlineEvaluator(args.model_name_or_path)
    elif args.benchmark == "PairWiseMercury":
        evaluator = PairWiseEvaluator(args.model_name_or_path, args.do_generate)
    elif args.benchmark == "DistributeWiseMercury":
        evaluator = DistributeWiseEvaluator(args.model_name_or_path, args.do_generate)
        
    if args.do_generate:
        output = evaluator.generate(num_samples_per_task=args.samples)
    if args.do_evaluate:
        output = evaluator.evaluate(num_samples_per_task=args.samples)
        
    
    print("Bingo!")