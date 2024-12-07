from _modules import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import util
import model_util
from configs import llm_config, fed_config

util.clear_memory()
util.print_torch_cuda_info()
load_dotenv()

login(token=os.getenv('HF_HUB_TOKEN'))

RUN_DEVICE = util.get_device()
log.info(f"Running on Device: {RUN_DEVICE}")

def train_model(model, tokenizer, train_dataset, adapter_dir, output_dir, decription):
  log.info(f"Training {decription} model")
  log.info(f"Model: {llm_config.model_id}")
  
  log.info(f"Model Configuration: lora_rank={llm_config.lora_r}, target_modules={llm_config.LORA_TARGET_MODULES}")
  
  peft_config = LoraConfig(
    lora_alpha=llm_config.lora_alpha,
    lora_dropout=llm_config.lora_dropout,
    r=llm_config.lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=llm_config.LORA_TARGET_MODULES
  )
    
  model = prepare_model_for_kbit_training(model)
  log.info(f"Model prepared for int8 training")
  
  model = get_peft_model(model, peft_config)
  
  total_params = model.num_parameters()
  trainable_params = model.num_parameters(only_trainable=True)
  log.info(f"Model Parameters: {total_params} total, {trainable_params} trainable")
  log.info(f"Model prepared with peft config")
  
  model.save_pretrained(adapter_dir)
  log.info(f"Model saved at {adapter_dir}")

  return trainable_params, "NA", "NA"

def run_experiment(despription, model_type, output_dir):
  if not os.path.exists(f"{output_dir}{model_type}"):
    os.makedirs(f"{output_dir}{model_type}")
    log.info(f"Created directory: {output_dir}{model_type}")
  else:
    log.info(f"Directory already exists: {output_dir}{model_type}")
    
  model, tokenizer = model_util.prepare_model(llm_config.model_id)
  
  test_dataset = load_from_disk(fed_config.test_dataset_dir)
  log.info(f"Test Dataset loaded from {fed_config.test_dataset_dir}")
  
  train_dataset = load_from_disk(fed_config.train_dataset_dir)
  log.info(f"Train Dataset loaded from {fed_config.train_dataset_dir}")
  
  projection_matrix = ''.join([module[0] for module in llm_config.LORA_TARGET_MODULES])
  adapter_dir = f"{output_dir}{model_type}/TrD{len(train_dataset)}Ada_W{projection_matrix}_r{llm_config.lora_r}/"
  
  if not os.path.exists(adapter_dir):
    os.makedirs(adapter_dir)
    log.info(f"Created directory: {adapter_dir}")
  else:
    log.info(f"Directory already exists: {adapter_dir}")
  
  trainable_params, before_time_str, after_time_str = train_model(model, tokenizer, train_dataset, adapter_dir, output_dir, adapter_dir)
  
  csv_params = {
    "model_type": model_type,
    "round": 0,
    "trainable_params": trainable_params,
    "before_time_str": before_time_str,
    "after_time_str": after_time_str
  }    
  
  evaluation_results = util.evaluate(model, model_type, tokenizer, test_dataset, train_dataset, csv_params, adapter_dir + "evaluation_results.csv")
  if evaluation_results:
    log.info(f"Experiment Completed")
  else:
    log.error(f"Experiment Failed")   

if __name__ == "__main__":
  start_time = datetime.datetime.now()
  log.info(f"Experiment started at {start_time}")
  log.info(f"*"*50)
  
  run_experiment("Federated Model, Base Model", "Fed0", fed_config.output_dir)
  
  log.info(f"*"*50)
  end_time = datetime.datetime.now()
  
  log.info(f"Experiment ended at {end_time}")

