import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import flwr as fl
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from datasets import load_dataset
from flwr.client.mod import fixedclipping_mod
from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping

from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import argparse

from utils.utils import *

cfg = get_config("centralized_full")

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default=None)
parser.add_argument("--dataset_name", type=str, default=None)
args = parser.parse_args()

modelFolderName = cfg.model.name
if args.base_model_path is not None:
    cfg.model.name = args.base_model_path
if args.dataset_name is not None:
    cfg.dataset.name = args.dataset_name

print_config(cfg)

with open(cfg.dataset.path.format(cfg.dataset.name), "r") as file:
    dataset = Dataset.from_list(json.load(file))

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(
    cfg.model.name,
    use_fast=cfg.model.use_fast_tokenizer,
    # padding_side=cfg.train.padding_side,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = (
        tokenizer.bos_token if tokenizer.padding_side == "left" else tokenizer.eos_token
    )
print(f"pad_token_id: {tokenizer.pad_token_id}")

save_path = f"./models/centralized/{modelFolderName}/{cfg.dataset.name}/{(datetime.now()).strftime("%Y%m%d%H%M%S")}"

model = get_model(cfg.model)

CUDA = torch.cuda.is_available()
desirable_weight = 1
undesirable_weight = 1
if dataset is not None:
    numDesirable = sum(dataset["label"]) * 3
    numUndesirable = (dataset.num_rows - numDesirable / 3) * 4
    if numDesirable < numUndesirable:
        desirable_weight = numUndesirable / numDesirable
    else:
        undesirable_weight = numDesirable / numUndesirable
print(f"numDesirable: {sum(dataset["label"])}")
print(f"numUndesirable: {dataset.num_rows - sum(dataset["label"])}")
print(f"desirable_weight: {desirable_weight}")
print(f"undesirable_weight: {undesirable_weight}")

training_argumnets = KTOConfig(
    **cfg.train.training_arguments,
    output_dir=save_path,
    desirable_weight=desirable_weight,
    undesirable_weight=undesirable_weight,
)

trainer = KTOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_argumnets,
    train_dataset=dataset,
)

# Do training
results = trainer.train()

model.save_pretrained(save_path)
