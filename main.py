import hydra
import pickle
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import flwr as fl

from dataset import load_federated_data
from client import generate_client_fn, generate_client_fn_kd
from server import get_evaluate_fn, get_on_fit_config, weighted_average
from model import Net, get_parameters
from strategy import HeteroLoRA, HeteroLoraKD

from utilis import fit_config

import torch
import time


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    ## step 1: load dataset
    # only evaluate at the server side
    trainloaders, testloader = load_federated_data(cfg.num_clients, cfg.checkpoint)

    ## step 2 define client
    # this function needs every details about clients
    # If one wants to validate at the client side, an extra parameter valloaders needs to be added
    # client_fn, global_net, heterogeneous_nets = generate_client_fn(trainloaders, cfg.num_classes, cfg.checkpoint, cfg.r, cfg.hetero, cfg.kd)
    client_fn, heterogeneous_nets = generate_client_fn_kd(trainloaders, testloader, cfg.num_classes, cfg.checkpoint, cfg.r)


    # strategy = HeteroLoRA(Net, 
    #                       fraction_fit=1.0,
    #                       min_fit_clients=2,
    #                       min_available_clients=2,
    #                       evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    #                       initial_parameters=fl.common.ndarrays_to_parameters(params), # set the initial parameter on the server side
    #                       r_values=cfg.r,
    #                       hetero=cfg.hetero,
    #                       hetero_net=heterogeneous_nets,
    #                       padding_strategy=cfg.padding_strategy) 

    
    strategy = HeteroLoraKD(Net,
                            fraction_fit=1.0,
                            min_fit_clients=2,
                            min_available_clients=2,
                            # evaluate_fn=get_evaluate_fn(Net, cfg.num_classes, testloader),
                            r_values=cfg.r,
                            hetero_net=heterogeneous_nets,
                            on_fit_config_fn = fit_config,
                            )

    ## step 4: start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={'num_cpus': 2, 'num_gpus': 1.0},
    )



if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"training spends {(end_time-start_time):.2f} seconds")