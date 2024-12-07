from _modules import *
from configs import llm_config, fed_config

def read_adapter(adapter_dir):
  model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_dir,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=llm_config.device_map
  )
  log.info(f"Adapter reloaded from {adapter_dir}")
  return model

def get_lora_states(adapter):
  adapter_state = adapter.state_dict()
  lora_return = {}
  
  for key in adapter_state:
    if "lora_" in key:
      lora_return[key] = adapter_state[key]
      
      bias_name = key.split("lora_")[0] + "bias"
      if bias_name in adapter_state:
        lora_return[bias_name] = adapter_state[bias_name]
        
  return lora_return

def FedAvgIT(num_clients, adapter_locations, trainrows_details):
  if num_clients <1:
      print("error: FedAvg can only be applied on 1+ clients")
      return None
  elif num_clients ==1:
      return read_adapter(adapter_locations[0])
  
  totalrows = sum(trainrows_details)
  normalized_trainrows = [value / totalrows for value in trainrows_details]
  
  lora_adapters = []
  for i in range(0, num_clients):
      this_adapter = read_adapter(adapter_locations[i]) 
      this_lora = get_lora_states(this_adapter)
      del this_adapter
      lora_adapters.append(this_lora)
  
  avg_adapter_state_dict = OrderedDict()
  
  for key in lora_adapters[0].keys():
      avg_adapter_state_dict[key] = sum(
          lora_adapters[i][key] * normalized_trainrows[i]
          for i in range(num_clients)
      )
  del lora_adapters
  avg_loraadapter = read_adapter(adapter_locations[0]) 
  for key in avg_loraadapter.state_dict().keys():
      if key not in avg_adapter_state_dict.keys():
          avg_adapter_state_dict[key] = avg_loraadapter.state_dict()[key]

  avg_loraadapter.load_state_dict(avg_adapter_state_dict)
  return avg_loraadapter

def FedAvg(num_clients, adapter_dirs, trainrows_details):
  if num_clients < 1:
    log.error("No clients to federate")
    return None

  totalrows = sum(trainrows_details)
  normalized_trainrows = [trainrow / totalrows for trainrow in trainrows_details]

  lora_adapters = []
  for i in range(num_clients):
    x_adapter = read_adapter(adapter_dirs[i])
    x_lora = get_lora_states(x_adapter)
    del x_adapter
    lora_adapters.append(x_lora)

  avg_adapter_state_dict = OrderedDict()
  key_prefix = []
  for key in lora_adapters[0].keys():
    parts = key.split("lora_")
    if len(parts) > 1:
      prefix = parts[0]
      if prefix not in key_prefix:
        key_prefix.append(prefix)

  def MatrixDecomposition(org_matrix, B, A):
    learning_rate = 0.001
    max_iterations = 5000

    prev_error = torch.tensor(float('inf'))
    itertion = 0

    while itertion < max_iterations:
      error = org_matrix - B @ A
      error_norm = torch.norm(error)

      if torch.abs(error_norm - prev_error) < 1e-6:
        log.info(f"Converged(sufficiently) at iteration {itertion}")
        break

      grad_B = -2 * error @ torch.transpose(A, 0, 1)
      grad_A = -2 * torch.transpose(B, 0, 1) @ error

      B -= learning_rate * grad_B
      A -= learning_rate * grad_A

      prev_error = error_norm
      itertion += 1

    log.info(f"Matrix Decomposition completed in {itertion} iterations, final error: {error_norm}")
    return B, A

  highest_client = normalized_trainrows.index(max(normalized_trainrows))
  log.info(f"Client {highest_client} has the highest train rows")

  for key in key_prefix:
    keyA = key + "lora_A" + ".default.weight"
    keyB = key + "lora_B" + ".default.weight"

    AverageBA = sum(
      lora_adapters[i][keyB] @ lora_adapters[i][keyA] * normalized_trainrows[i]
      for i in range(num_clients)
    )

    B, A = MatrixDecomposition(AverageBA, lora_adapters[highest_client][keyB], lora_adapters[highest_client][keyA])
    avg_adapter_state_dict[keyA] = A
    avg_adapter_state_dict[keyB] = B

  avg_lora_adapter = read_adapter(adapter_dirs[highest_client])

  for key in avg_lora_adapter.state_dict().keys():
    if key not in avg_adapter_state_dict.keys():
      avg_adapter_state_dict[key] = avg_lora_adapter.state_dict()[key]

  avg_lora_adapter.load_state_dict(avg_adapter_state_dict)
  return avg_lora_adapter