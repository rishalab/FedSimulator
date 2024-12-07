# Silence warnings
import warnings
warnings.filterwarnings("ignore")

import sys
import os
import io
import contextlib
import re
import csv
import json
import datetime
import random
import logging as log
log.basicConfig(level=log.INFO, 
    format='%(asctime)s.%(msecs)03d %(levelname)s : %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

import torch
import pandas as pd
import gc
import subprocess
import transformers

from io import StringIO
from random import randrange
from datasets import load_dataset
from datasets import Dataset
from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, LlamaTokenizer, LlamaForCausalLM, Trainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, get_peft_model_state_dict, AutoPeftModelForCausalLM, prepare_model_for_int8_training
from trl import SFTTrainer
from collections import OrderedDict
from huggingface_hub import login
from dotenv import load_dotenv
from torch import nn

# Evaluation
from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score