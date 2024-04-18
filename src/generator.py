# coding: utf-8

# Strik on the assigned GPU.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = 'false'
os.environ['HF_HOME'] = '/home/mingzhe/hf_cache'

