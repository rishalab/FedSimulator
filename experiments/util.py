import gc
import glob
import shutil
from configs import fed_config, llm_config
import util
from _modules import *
import torch
import logging

log = logging.getLogger(__name__)

def setup_logger(log_file):
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
  
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def print_torch_cuda_info():
    log.info("*** Torch Cuda Information ***")
    log.info("Torch Version: %s", torch.__version__)
    log.info("CUDA Available: %s", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        log.info("CUDA Device Count: %s", torch.cuda.device_count())
        log.info("CUDA Device Name: %s", torch.cuda.get_device_name())
        log.info("CUDA Current Device: %s", torch.cuda.current_device())
        log.info("CUDA Device Capability: %s", torch.cuda.get_device_capability())
        log.info("CUDA Device Memory: %s GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

def format_instruction(row):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ### Instruction:
    {"generate docstring for the below python function"}
    ### Input:
    {row["function"]}
    ### Reponse:
    {row["docstring"]}"""

def generate_docstring(model, row, tokenizer):
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ### Instruction:
    {"generate docstring for the below python function"}
    ### Input:
    {row['function']}
    ### Reponse:
    """
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=0.5)
    
    generate_docstring = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
    return generate_docstring, row['docstring']

def evaluate_docstring(model, model_type, tokenizer, test_dataset):
    rouge = Rouge()

    generated, actual = [], []
    total_count = len(test_dataset)
    log.info(f"Total Test Data: {total_count}")
    
    for i, row in enumerate(test_dataset):
      generated_docstring, actual_docstring = generate_docstring(model, row, tokenizer)
      generated.append(generated_docstring), actual.append(actual_docstring)

      if i % 2 == 0:
        log.info(f"Evaluation: {i}/{total_count}")
    
    log.info("Generating Docstrings Completed")
    log.info("Evaluating Docstrings")
        
    # Health Check for Generated Docstrings
    for idx, content in enumerate(generated):
      if not content.strip():
        generated[idx] = "NO_OUTPUT"

    for idx, content in enumerate(actual):
      if not content.strip():
        actual[idx] = "NO_INPUT"
        generated[idx] = "SO_NO_OUTPUT"
    
    # Save as array in a json file
    model_prefix = model_type.lower().replace(" ", "_")
    with open(model_prefix + "_generated_docstrings.json", "w") as f:
      json.dump(generated, f)
      
    with open(model_prefix + "_actual_docstrings.json", "w") as f:
      json.dump(actual, f)

    # Load from json file
    log.info(f"Generated(size): {len(generated)}")
    log.info(f"Actual(size): {len(actual)}")
    
    try:
      rouge_scores = rouge.get_scores(generated, actual, avg=True)
      bleu_score = corpus_bleu([[act.split()] for act in actual], [gen.split() for gen in generated])
      meteor_scores = [meteor_score([act.split()], gen.split()) for act, gen in zip(actual, generated)]
      average_meteor_score = sum(meteor_scores) / len(meteor_scores)
    except Exception as e:
      log.error("Error Evaluating Docstrings")
      log.error(e)
      return {
        "BLEU": "Invalid",
        "METEOR": "Invalid",
        "ROUGE": "Invalid"
      }

    log.info(f"Rouge Scores: {rouge_scores}")
    log.info(f"Bleu Score: {bleu_score}")
    log.info(f"Meteor Score: {average_meteor_score}")

    evaluation_results = {
      "BLEU": bleu_score,
      "METEOR": average_meteor_score,
      "ROUGE": rouge_scores
    }

    log.info("Evaluation Completed")
    return evaluation_results


def evaluate(model, model_type, tokenizer, test_dataset, train_dataset, csv_params, csv_path):
  log.info("Evaluating model")
  log.info(f"Test Dataset Size: {len(test_dataset)}")
  log.info(f"Train Dataset Size: {len(train_dataset)}")

  
  metric_results = evaluate_docstring(model, model_type, tokenizer, test_dataset)
  column_headers = ["Data", "TrainDataSize", "ModelType", "Round","BaseModelName", "TrainableParams", "TargetModules", "RankLoRAModule", "StartTime", "EndTime", "NumberEpochs", "EvalDataSize", "C-BLEU", "METEOR", "ROUGE"]

  try:
    if not os.path.exists(csv_path):
      with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(column_headers)

    with open(csv_path, "a") as f:
      writer = csv.writer(f)
      writer.writerow([
        fed_config.train_dataset_dir,
        len(train_dataset),
        csv_params["model_type"],
        csv_params["round"],
        llm_config.model_id,
        csv_params["trainable_params"],
        llm_config.LORA_TARGET_MODULES,
        llm_config.lora_r,
        csv_params["before_time_str"],
        csv_params["after_time_str"],
        llm_config.num_train_epochs,
        len(test_dataset),
        metric_results["BLEU"],
        metric_results["METEOR"],
        metric_results["ROUGE"]
      ])
  except Exception as e:
    log.error(f"Error saving results to {csv_path}")
    log.error(e)
    return False

  log.info(f"Results saved to {csv_path}")
  return True


def delete_directory(directory):
    merged_model_dir = directory
    try:
        shutil.rmtree(merged_model_dir)
    except FileNotFoundError:
        log.info(f"The del_directory {merged_model_dir} does not exist.")
    except Exception as e:
        log.info(f"An error occurred while deleting the contents of {merged_model_dir}: {str(e)}")

def combine_evaluation_results(final_merged_csv):
  csv_files = glob.glob("outputs/Fedn/*/server/Ada_Impr/evaluation_results.csv")
  log.info(f"Total CSV Files: {len(csv_files)}")
  
  combined_csv = pd.concat([pd.read_csv(f) for f in csv_files])
  combined_csv = combined_csv.sort_values(by=['Round'])
  combined_csv.to_csv(final_merged_csv, index=False, encoding='utf-8-sig')
  log.info(f"Combined CSV saved to {final_merged_csv}")