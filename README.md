# LLM2FedLLM - Simulating LLMs in Federated Setting

## Description
This repository contains the code for simulating the performance of Language Model Models (LLMs) in a federated learning setting. The code is designed to simulate the performance of LLMs in a federated learning setting using the FedAvg algorithm. The code is designed to be modular and can be easily extended to other LLMs and federation algorithms.

## Setup
### Install dependencies
```bash
git clone https://github.com/rishalab/LLM2FedLLM.git
cd LLM2FedLLM
pip install -r requirements.txt
```

### Configure the project
- Place your HuggingFace Access Token in the .env file.
- Modify the experiments/configs/fed_config.py and experiments/configs/llm_config.py files to set the desired configurations for the federated learning setting and LLM experiments respectively.
- Unzip the datasets.zip file in the experiments directory to get the sample datasets.

## Run
- Make sure to have a snapshots of the experiments/output directory to prevent overwriting & appending of the results.
- Clear the output directory if you want to start fresh.
- Make sure you have placed the datasets in the datasets directory and edited the config files accordingly.
Refer to example datasets in the experiments/datasets directory. 
```bash
experiments/datasets/
---> code_docstring_corpus_data_test/
---> code_docstring_corpus_data_train/
```
For FL client split, split the data into required no of clients based on the unique marker (such as project, task, etc) in the dataset.
```bash
experiments/datasets/
---> client_datasets/client_*/
```

- General command to run the experiments.
```bash
cd experiments
python3 <*>.py
```

### Central Training
- Run the central training script to train & evaluate the LLM on the central server.
```bash
python3 central.py
```

### Federated Learning(Base Model (fed0))
- Run the federated learning script to train & evaluate the LLM in the federated setting.
```bash
python3 fed0.py
```

### Federated Learning (n round) (Fedn)
- Edit the fed_config.py file to set the desired number of rounds.
- Run the federated learning script to train & evaluate the LLM in the federated setting using FedAvg.
```bash
python3 fedn.py
```

By default the experiment will run the evaluation for central, fed0, fedn setting and generated the results in the output directory and annecdotal results in the experiments directory.

### Federated Learning (Fedn Best Evaluation)
- Edit the run_fedNBest to True and set the id of the best model to be used for the federated learning evaluation.
```bash
python3 fedn.py
```

### Federated Learning (Fedn Client Evaluation)
- Edit the run_clientEvaluation to True.
```bash
python3 fedn.py
```


