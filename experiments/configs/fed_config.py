## Federated Learning Configuration

# Number of clients to use for federated learning
num_clients = 3
# Number of rounds to run federated learning
num_rounds = 20

# Training args
num_train_rows = 26980
train_dataset_dir = "datasets/code_docstring_corpus_data_train/"
test_dataset_dir = "datasets/code_docstring_corpus_data_test/"
dataset_ratio_split4clients = (7332,2898,16750)
client_datasets_dir = "client_datasets"
history_file_path = '../HetroCS_Laama2LoRA.csv'

output_dir = "outputs/"

run_fedNBest = False
fedNBestRound = 9 # Change based on the round to run FedNBest

run_clientEvaluation = False

run_fedN = True
fed_type = "FedAvgIT"
peft_type = "QLORA"

