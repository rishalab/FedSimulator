import os
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()
    
setup(
  name="llm2fedllm",
  version='1.0.0',
  description="A simulator for comparing federated approach to centralized approach in finetuning LLM",
  author="Jahnavi K, Siddhartha G",
  author_email="cs22s503@iittp.ac.in, cs20b040@iittp.ac.in",
  install_requires=required,
)