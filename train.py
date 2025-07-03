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

cfg = get_config("federated_full")

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default=None)
parser.add_argument("--num_rounds", type=str, default=None)
parser.add_argument("--dataset_name", type=str, default=None)
args = parser.parse_args()

modelFolderName = cfg.model.name
if args.base_model_path is not None:
    cfg.model.name = args.base_model_path
if args.num_rounds is not None:
    cfg.flower.num_rounds = int(args.num_rounds)
if args.dataset_name is not None:
    cfg.dataset.name = args.dataset_name

print_config(cfg)

with open(cfg.dataset.path.format(cfg.dataset.name), "r") as file:
    datasets = json.load(file)
cfg.flower.num_clients = len(datasets.keys())
print(cfg.flower.num_clients)

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

save_path = f"./models/{modelFolderName}/{cfg.dataset.name}/{(datetime.now()).strftime("%Y%m%d%H%M%S")}"
client = fl.client.ClientApp(
    client_fn=gen_client_fn(
        datasets,
        tokenizer,
        cfg.model,
        cfg.train,
        save_path,
    ),
    # mods=[fixedclipping_mod] # For Differential Privacy
)


def server_fn(context: Context):

    # Define the Strategy
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=cfg.flower.num_clients,  # total clients
        fraction_fit=cfg.flower.sample_clients
        / cfg.flower.num_clients,  # ratio of clients to sample
        fraction_evaluate=0.0,  # No federated evaluation
        # A (optional) function used to configure a "fit()" round
        on_fit_config_fn=get_on_fit_config(),
        # A (optional) function to aggregate metrics sent by clients
        fit_metrics_aggregation_fn=fit_weighted_average,
        # A (optional) function to execute on the server after each round.
        # In this example the function only saves the global model.
        evaluate_fn=get_evaluate_fn(
            cfg.model, cfg.train.save_every_round, cfg.flower.num_rounds, save_path
        ),
    )

    # # Add Differential Privacy
    # sampled_clients = cfg.flower.num_clients*strategy.fraction_fit
    # strategy = DifferentialPrivacyClientSideFixedClipping(
    #     strategy,
    #     noise_multiplier=cfg.flower.dp.noise_mult,
    #     clipping_norm=cfg.flower.dp.clip_norm,
    #     num_sampled_clients=sampled_clients
    # )

    # Number of rounds to run the simulation
    num_rounds = cfg.flower.num_rounds
    config = fl.server.ServerConfig(num_rounds=num_rounds)

    return fl.server.ServerAppComponents(strategy=strategy, config=config)


server = fl.server.ServerApp(server_fn=server_fn)

client_resources = dict(cfg.flower.client_resources)
fl.simulation.run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=cfg.flower.num_clients,
    backend_config={"client_resources": client_resources, "init_args": backend_setup},
    verbose_logging=True,
)
