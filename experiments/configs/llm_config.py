## Model Configuration

# Model that will be used for the experiment, ref. to huggingface.co
# model_id = "meta-llama/Llama-3.2-1B"
model_id = "NousResearch/Llama-2-7b-hf"
# Load the entire model on the first GPU on the machine
device_map = {"": 0}


## Bitsandbytes Parameters

# Use 4-bit quantization
use_4bit = False
# Use float32 for compute_dtype
bnb_4bit_compute_dtype = "float32"
# Use quantization type
bnb_4bit_quant_type = "fp4" # fp4 or nf4
# Use nested quantization for 4-bit base model
use_double_nested_quant = False


## QLora Parameters

# Attention dimension for LoRA
lora_r = 8
# Alpha parameter for LoRA
lora_alpha = 16
# Dropout for LoRA
lora_dropout = 0.1
# Modules to apply LoRA
LORA_TARGET_MODULES = ["q_proj", "k_proj"]


## Training Argument Parameters

# No of epochs to train the model
num_train_epochs = 1
# Enable fp16/bf16 training (bf16 for A100 GPUs)
fp16 = True
bf16 = False
# Batch size per GPU for training
per_device_train_batch_size = 6 
# Accumulation steps for training
gradient_accumulation_steps = 1
# Checkpointing for gradient
gradient_checkpointing = True
# Maximum gradient norm for clipping
max_grad_norm = 0.3
# Optimizer to use
optim = "adamw_torch"
# Initial learning rate for optimizer
learning_rate = 2e-4
# Weight decay for optimizer
weight_decay = 0.001
# Scheduler to use
lr_scheduler_type = "cosine"
# No of training steps (overrides num_train_epochs)
max_steps = -1
# Ratio of steps for a linear warmup
warmup_ratio = 0.03
# Group sequences for faster training
group_by_length = False
# Save model checkpoints after n steps
save_steps = -1
# Log every x updates steps
logging_steps = 25

disable_tqdm = True
max_split_size_mb = 2000


## SFTT Trainer Parameters

# Maximum sequence length for SFTT
max_seq_length = 2048
# Pack multiple sequences into one tensor
packing = True