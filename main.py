import hydra
import pickle
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import flwr as fl

from dataset import load_federated_data
from client import generate_client_fn
from server import get_evaluate_fn, get_on_fit_config, weighted_average
from model import Net

import torch


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    ## step 1: load dataset
    #TODO: Generalize the load_dataset function, make it can load different datasets and set different iid/non-iid settings
    # only evaluate at the server side
    trainloaders, testloader = load_federated_data(cfg.num_clients, cfg.checkpoint)

    ## step 2 define client
    # this function needs every details about clients
    # If one wants to validate at the client side, an extra parameter valloaders needs to be added
    client_fn = generate_client_fn(trainloaders, cfg.num_classes, cfg.checkpoint, cfg.r, cfg.hetero)

    ## step 3 define strategy
    #TODO: custom a new strategy, especially the aggregation strategy, where can be extended from def aggregate_fit()
    strategy = fl.server.strategy.FedAvg(fraction_fit=1.0,
                                         fraction_evaluate=0.5,
                                         evaluate_fn=get_evaluate_fn(cfg.num_classes,
                                                                     testloader),
                                        #  evaluate_metrics_aggregation_fn=weighted_average,
                                         )
    
    ## step 4: start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={'num_cpus': 2, 'num_gpus': 1.0},
    )

    ## step 6: save results
    # save_path = HydraConfig.get().runtime.output_dir
    # results_path = Path(save_path) / 'result.pkl'

    # results = {'history': history, 'anythingelse': "here"}

    # with open(str(results_path), 'wb') as h:
    #     pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()