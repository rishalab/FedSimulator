from _modules import *
from configs import llm_config, fed_config

def prepare_model(model_id):
  model_kwargs = {
    "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
    "device_map": "auto",
    "return_dict": True
  }
  
  model = AutoModelForCausalLM.from_pretrained(llm_config.model_id, **model_kwargs)
  tokenizer = AutoTokenizer.from_pretrained(llm_config.model_id)
  
  # Freeze original weights except for layer norms
  for param in model.parameters():
    param.requires_grad = False
    if param.ndim == 1:
      param.data = param.data.to(torch.float32)
    
  class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(x.dtype)
  
  model.lm_head = CastOutputToFloat(model.lm_head)

  model.gradient_checkpointing_enable()
  model.enable_input_require_grads()
  
  # Prep tokenizer
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    log.info(f"Tokenizer pad token set to eos token")
    
  tokenizer.padding_side = "right"
  model.generation_config.pad_token_id = tokenizer.eos_token_id
  
  return model, tokenizer

def get_merged_model(adapter_dir):
  model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_dir,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=llm_config.device_map
  )
  model.gradient_checkpointing_enable()
  model.enable_input_require_grads()
  
  log.info(f"Model reloaded from {adapter_dir}")
  merged_model = model.merge_and_unload()
  del model

  return merged_model


def getModelToBeTrainedLORA(prev_model_dir):
  merged_model_dir = prev_model_dir + "../merged_model_Impr/"
  
  peft_config = LoraConfig(
    lora_alpha=llm_config.lora_alpha,
    lora_dropout=llm_config.lora_dropout,
    r=llm_config.lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=llm_config.LORA_TARGET_MODULES
  )
  
  model = AutoModelForCausalLM.from_pretrained(
    merged_model_dir,
    use_cache=False,
    torch_dtype=torch.float16,
    device_map=llm_config.device_map
  )
  model.gradient_checkpointing_enable()
  model.enable_input_require_grads()
  
  model.config.pretraining_tp = 1
  model = prepare_model_for_kbit_training(model)


  model = get_peft_model(model, peft_config)
  log.info(f"Model reloaded from {merged_model_dir}")
  return model

def getModelToBeTrainedQLORA(prevmodel_dir):
    compute_dtype = getattr(torch, llm_config.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=llm_config.use_4bit,
        bnb_4bit_use_double_quant=llm_config.use_double_nested_quant,
        bnb_4bit_quant_type=llm_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype
    )

    merged_model_dir = prevmodel_dir + "../merged_model_Impr/"
  
    peft_config = LoraConfig(
      lora_alpha=llm_config.lora_alpha,
      lora_dropout=llm_config.lora_dropout,
      r=llm_config.lora_r,
      bias="none",
      task_type="CAUSAL_LM",
      target_modules=llm_config.LORA_TARGET_MODULES
    )
  
    model = AutoModelForCausalLM.from_pretrained(
      merged_model_dir,
      quantization_config=bnb_config,
      use_cache=False,
      torch_dtype=torch.float16,
      device_map=llm_config.device_map
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, peft_config)
    log.info(f"Model reloaded from {merged_model_dir}")
    return model