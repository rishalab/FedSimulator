from _modules import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import util
import model_util
from configs import llm_config, fed_config

from configs.FedAggr import FedAvg, FedAvgIT

util.clear_memory()
util.print_torch_cuda_info()
load_dotenv()

login(token=os.getenv('HF_HUB_TOKEN'))

RUN_DEVICE = util.get_device()
log.info(f"Running on Device: {RUN_DEVICE}")

# Default Aggregation type is FedWithDecomp
FedAvgBasedOnType = None
getModelBasedOnConfig = None

peft_config = LoraConfig(
  lora_alpha=llm_config.lora_alpha,
  lora_dropout=llm_config.lora_dropout,
  r=llm_config.lora_r,
  bias="none",
  task_type="CAUSAL_LM",
  target_modules=llm_config.LORA_TARGET_MODULES
)
  
"""
  Federation Logic
"""
def FedAvgWithReload(num_clients, adapter_dirs, datasetksplit):
  federated_model = FedAvgBasedOnType(num_clients, adapter_dirs, datasetksplit)
  return federated_model

def train_save_model(model, tokenizer, output_dir, adapter_dir, dataset):
  args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=llm_config.num_train_epochs,
    per_device_train_batch_size=llm_config.per_device_train_batch_size,
    gradient_accumulation_steps=llm_config.gradient_accumulation_steps,
    gradient_checkpointing=llm_config.gradient_checkpointing,
    optim=llm_config.optim,
    logging_steps=llm_config.logging_steps,
    save_strategy="no",
    learning_rate=llm_config.learning_rate,
    weight_decay=llm_config.weight_decay,
    fp16=llm_config.fp16,
    bf16=llm_config.bf16,
    max_grad_norm=llm_config.max_grad_norm,
    warmup_ratio=llm_config.warmup_ratio,
    group_by_length=llm_config.group_by_length,
    disable_tqdm=llm_config.disable_tqdm,
    lr_scheduler_type=llm_config.lr_scheduler_type,
    seed=42
  )
  
  data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    pad_to_multiple_of=8,
    return_tensors="pt",
    padding=True
  )
  
  peft_config = LoraConfig(
    lora_alpha=llm_config.lora_alpha,
    lora_dropout=llm_config.lora_dropout,
    r=llm_config.lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=llm_config.LORA_TARGET_MODULES
  )
  
  trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=llm_config.max_seq_length,
    tokenizer=tokenizer,
    packing=llm_config.packing,
    formatting_func=util.format_instruction,
    args=args,
    data_collator=data_collator
  )
  
  before_time = datetime.datetime.now()
  before_time_str = before_time.strftime("%Y-%m-%d %H:%M:%S")
  log.info(f"Training started at {before_time_str}")
  
  trainer.train()
  
  after_time = datetime.datetime.now()
  after_time_str = after_time.strftime("%Y-%m-%d %H:%M:%S")
  log.info(f"Training ended at {after_time_str}")
  
  model.save_pretrained(adapter_dir)
  log.info(f"Model saved at {adapter_dir}")
  
  del trainer
  del model
  
  return before_time_str, after_time_str

def update_basemodel_inAdaCfg(adapter_dir):
  config_file_path = os.path.join(adapter_dir, "adapter_config.json")
  
  with open(config_file_path, "r") as f:
    config_data = json.load(f)
    
  config_data["base_model_name_or_path"] = llm_config.model_id
  
  with open(config_file_path, "w") as f:
    json.dump(config_data, f, indent=4)
    
  log.info(f"Base model updated in adapter config file")

def check_base_model(this_round):
  directory_path = "outputs/Fedn/" + "round" + str(this_round) + "/server/merged_model_Impr/"
  adapter_dir = "outputs/Fedn/" + "round" + str(this_round) + "/server/Ada_Impr/"
  
  if os.path.exists(directory_path):
    log.info(f"Base model exists at {directory_path}")
  else:
    log.info(f"Base model does not exist at {directory_path}")
    if this_round == 0:
      model, tokenizer = model_util.prepare_model(llm_config.model_id)
      model = prepare_model_for_kbit_training(model)
      model = model_util.get_peft_model(model, peft_config)
      model.save_pretrained(adapter_dir)
      
      model = model_util.get_merged_model(adapter_dir)
      model.save_pretrained(directory_path)
      log.info(f"Base model saved at {directory_path}")
    else:
      log.error(f"Base model does not exist at {directory_path}")
      
      model = model_util.get_merged_model(adapter_dir)
      model.save_pretrained(directory_path)
      
      log.info(f"Round model saved at {directory_path}")

def set_global_params(fed_type, peft_type):
  global FedAvgBasedOnType, getModelBasedOnConfig
  if fed_type == "FedAvgIT":
    FedAvgBasedOnType = FedAvgIT
  else:
    FedAvgBasedOnType = FedAvg
    
  if peft_type == "QLORA":
    getModelBasedOnConfig = model_util.getModelToBeTrainedQLORA
  else:
    getModelBasedOnConfig = model_util.getModelToBeTrainedLORA

def run_fedn_experiment(fed_type, peft_type):
  set_global_params(fed_type, peft_type)
  model, tokenizer = model_util.prepare_model(llm_config.model_id)

  train_dataset = load_from_disk(fed_config.train_dataset_dir)
  log.info(f"Train Dataset loaded from {fed_config.train_dataset_dir}")

  test_dataset = load_from_disk(fed_config.test_dataset_dir)
  log.info(f"Test Dataset loaded from {fed_config.test_dataset_dir}")

  num_training_points = len(train_dataset)
  log.info(f"Total training points: {num_training_points}")

  num_clients = fed_config.num_clients
  log.info(f"Number of clients: {num_clients}")

  # -------------------------------------------------------------------- #
  # Split the dataset into k parts and save them in client_datasets/
  # -------------------------------------------------------------------- #
  # random.seed(42)
  # random_values = [random.random() for i in range(num_clients)]
  # random_ratio = [value/sum(random_values) for value in random_values]
  # log.info(f"Random ratios: {random_ratio}")

  # datasetksplit = [int(value*num_training_points) for value in random_ratio]
  # datasetksplit[-1] += num_training_points - sum(datasetksplit)
  # log.info(f"Dataset split: {datasetksplit}")
  
  # # Split the dataset into k parts and save them in client_datasets/
  # for i in range(num_clients):
  #   start = sum(datasetksplit[:i])
  #   end = sum(datasetksplit[:i+1])
  #   client_dataset = train_dataset.select(range(start, end))
  #   client_dataset.save_to_disk(f"datasets/client_datasets/client_{i}/")
  #   log.info(f"Client {i} dataset saved at client_datasets/client_{i}/")

  # dataset_ratio_split4clients = datasetksplit
  # -------------------------------------------------------------------- #
  
  dataset_ratio_split4clients = fed_config.dataset_ratio_split4clients

  check_base_model(0)
  for this_round in range(1, fed_config.num_rounds+1):
    log.info("Beginning Round: " + str(this_round))
    
    adapter_locations = []
    before_times = []
    after_times = []
    
    lastround_server_dir = "outputs/Fedn/" + "round" + str(this_round-1) + "/server/" + "Ada_Impr/"
    
    for this_client in range(0, fed_config.num_clients):
      log.info(f"Round {this_round}, Client {this_client}")
      
      output_dir = "outputs/Fedn/" + "round" + str(this_round) + "/client" + str(this_client) + "/"
      adapter_dir = output_dir + "Ada_Impr/"
      adapter_locations.append(adapter_dir)
      
      model = getModelBasedOnConfig(lastround_server_dir)
      
      this_train_dataset_dir = "datasets/client_datasets/client_" + str(this_client) + "/"
      train_dataset = load_from_disk(this_train_dataset_dir)
      
      before_time, after_time = train_save_model(model, tokenizer, output_dir, adapter_dir, train_dataset)
      before_times.append(before_time)
      after_times.append(after_time)
      del model
      log.info(f"Round {this_round}, Client {this_client} completed")

    avg_ada = FedAvgWithReload(fed_config.num_clients, adapter_locations, dataset_ratio_split4clients)

    output_dir = "outputs/Fedn/" + "round" + str(this_round) + "/server/"
    adapter_dir = output_dir + "Ada_Impr/"
    avg_ada.save_pretrained(adapter_dir)
    log.info(f"Model saved at {adapter_dir}")
    del avg_ada

    merged_model = model_util.get_merged_model(adapter_dir)
    log.info(f"Model reloaded from {adapter_dir}")
    
    if this_round >= 2:
      util.delete_directory("outputs/Fedn/" + "round" + str(this_round-2) + "/server/merged_model_Impr/")
      log.info(f"Deleted round {this_round-2} merged model")
      
    merged_model.save_pretrained("outputs/Fedn/" + "round" + str(this_round) + "/server/merged_model_Impr/")
    log.info(f"Model saved at outputs/Fedn/round{this_round}/server/merged_model_Impr/")
    
    csv_params = {
      "model_type": "FedAvg",
      "round": this_round,
      "trainable_params": "NA",
      "before_time_str": before_times,
      "after_time_str": after_times
    }
    
    status = util.evaluate(merged_model, "fedn", tokenizer, test_dataset, train_dataset, csv_params, adapter_dir + "evaluation_results.csv")
    if status:
      log.info(f"Round {this_round} completed")
    else:
      log.error(f"Round {this_round} failed")

def run_fed_best_evaluation(round_id):
  model, tokenizer = model_util.prepare_model(llm_config.model_id)
  
  test_dataset = load_from_disk(fed_config.test_dataset_dir)
  log.info(f"Test Dataset loaded from {fed_config.test_dataset_dir}")
  
  train_dataset = load_from_disk(fed_config.train_dataset_dir)
  log.info(f"Train Dataset loaded from {fed_config.train_dataset_dir}")
  
  for i in range(0, round_id):
    check_base_model(i)
    
    if i > 1:
      util.delete_directory("outputs/Fedn/" + "round" + str(i-1) + "/server/merged_model_Impr/")
      log.info(f"Deleted round {i-1} merged model")
    
  adapter_dir = "outputs/Fedn/" + "round" + str(round_id) + "/server/Ada_Impr/"
  merged_model = model_util.get_merged_model(adapter_dir)
  
  csv_params = {
    "model_type": "FedAvg",
    "round": round_id,
    "trainable_params": "NA",
    "before_time_str": "NA",
    "after_time_str": "NA"
  }
  
  status = util.evaluate(merged_model, "fedn_best", tokenizer, test_dataset, train_dataset, csv_params, "outputs/Fedn/fedn_best" + "evaluation_results.csv")
  if status:
    log.info(f"Fed Best evaluation completed")
  else:
    log.error(f"Fed Best evaluation failed")
    
def run_all_clients_evaluation(round_id=1):
  model, tokenizer = model_util.prepare_model(llm_config.model_id)
  test_dataset = load_from_disk(fed_config.test_dataset_dir)
  log.info(f"Test Dataset loaded from {fed_config.test_dataset_dir}")
  
  check_base_model(0)
  
  for this_client in range(2, fed_config.num_clients):
    this_train_dataset_dir = "datasets/client_datasets/client_" + str(this_client) + "/"
    train_dataset = load_from_disk(this_train_dataset_dir)
    
    adapter_dir = "outputs/Fedn/" + "round" + str(round_id) + "/client" + str(this_client) + "/Ada_Impr/"
    merged_model = model_util.get_merged_model(adapter_dir)
    log.info(f"Model reloaded from {adapter_dir}")
    
    csv_params = {
      "model_type": "FedAvg",
      "round": round_id,
      "trainable_params": "NA",
      "before_time_str": "NA",
      "after_time_str": "NA"
    }
    
    if not os.path.exists("outputs/Fedn/fedn_clients"):
      os.makedirs("outputs/Fedn/fedn_clients")
    
    status = util.evaluate(merged_model, "fedn_client" + str(this_client), tokenizer, test_dataset, train_dataset, csv_params, "outputs/Fedn/fedn_clients/client_" + str(this_client) + "evaluation_results.csv")
    if status:
      log.info(f"Client {this_client} evaluation completed")
    else:
      log.error(f"Client {this_client} evaluation failed")

if __name__ == "__main__":
  start_time = datetime.datetime.now()
  log.info(f"Experiment started at {start_time}")
  log.info(f"*"*50)
  
  if fed_config.run_fedNBest:
    run_fed_best_evaluation(fed_config.fedNBestRound)
  
  if fed_config.run_clientEvaluation:
    run_all_clients_evaluation(1)
  
  if fed_config.run_fedN:
    run_fedn_experiment(fed_config.fed_type, fed_config.peft_type)
    
  log.info(f"*"*50)
  end_time = datetime.datetime.now()
  log.info(f"Experiment ended at {end_time}")



# -------------------------------------------------------------------- #
# run_all_clients_evaluation(1)
# run_fedn_experiment("FedAvgIT", "QLORA")
# util.combine_evaluation_results("outputs/Fedn/evaluation_results.csv")
# run_fed_best_evaluation(9)
# -------------------------------------------------------------------- #