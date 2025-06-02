from datasets import load_dataset
import re
import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from utils import labels, sample_data, format_instruction, make_instruction, make_instruction_test, extract_answer
from eval import mbti_accuracies
from pathlib import Path


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Custom dataset class for tokenized input
class CustomDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {key: val.squeeze(0) for key, val in self.data[idx].items()}
        item['labels'] = item['input_ids'].clone()
        return item

# Load dataset from CSV files
project_root = Path(__file__).resolve().parents[2]
full_datasets_dir = project_root / "full_datasets"

# Load the dataset
dataset = load_dataset("csv", data_files={
    "train": str(full_datasets_dir / "train_data_for_llm.csv"),
    "test": str(full_datasets_dir / "test_data_for_llm.csv")
})

# Sample a subset from full data
train_samples = sample_data(dataset["train"], 500, labels)
test_samples = sample_data(dataset["test"], 50, labels)

# Format each sample into instruction-style prompts
formatted_train = [format_instruction(sample) for sample in train_samples]
formatted_test = [format_instruction(sample) for sample in test_samples]

# Load model and tokenizer with quantization
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# Setup padding token
tokenizer.add_special_tokens({"pad_token": "<PAD>"})
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<PAD>")
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

# Tokenize formatted instructions
max_length = 300
tokenized_train = [
    tokenizer(make_instruction(item), truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    for item in formatted_train
]
tokenized_test = [
    tokenizer(make_instruction_test(item), truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    for item in formatted_test
]

# Calculate number of training steps
def calculate_max_steps(num_samples, num_epochs, batch_size, grad_accumulation_steps):
    effective_batch = batch_size * grad_accumulation_steps
    steps_per_epoch = num_samples // effective_batch
    return steps_per_epoch * num_epochs

num_samples = 1600
num_epochs = 1
batch_size = 2
grad_accumulation_steps = 4
max_steps = calculate_max_steps(num_samples, num_epochs, batch_size, grad_accumulation_steps)

# Configure LoRA for PEFT
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none"
)

# Wrap model with PEFT
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accumulation_steps,
    max_steps=400,
    warmup_steps=40,
    learning_rate=5e-4,
    fp16=True,
    logging_steps=10,
    output_dir="outputs",
    optim="paged_adamw_8bit",
    evaluation_strategy="steps",
    save_strategy="steps",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

# Trainer setup
trainer = Trainer(
    model=peft_model,
    train_dataset=CustomDataset(tokenized_train),
    eval_dataset=CustomDataset(tokenized_test),
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Disable cache for training
peft_model.config.use_cache = False

# Train the model
trainer.train()

# Generate predictions from fine-tuned model
list_resp = []
for i in tqdm(range(len(formatted_test))):
    text = make_instruction_test(formatted_test[i])
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = peft_model.generate(
        **inputs,
        max_new_tokens=5,
        pad_token_id=tokenizer.pad_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    list_resp.append(decoded)

# Extract predictions from raw generations
list_preds = [extract_answer(pred, labels) for pred in list_resp]
list_labels = [sample['output'].lower() for sample in formatted_test]

# Evaluate predictions
mbti_accuracies(list_labels, list_preds)
