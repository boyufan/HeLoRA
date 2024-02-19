
import hydra
import pickle
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import flwr as fl

from dataset import prepare_dataset
from client import generate_client_fn
from server import get_evaluate_fn, get_on_fit_config


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## Step 1
    print(OmegaConf.to_yaml(cfg))

    ## step 2: dataset
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size)

    ## step 3 define client
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    ## step 4 define strategy
    strategy = fl.server.strategy.FedAvg(fraction_fit=0.00001,
                                         min_fit_clients=cfg.num_clients_per_round_fit,
                                         fraction_evaluate=0.00001,
                                         min_evaluate_clients=cfg.num_clients_per_round_eval,
                                         min_available_clients=cfg.num_clients,
                                         on_fit_config_fn=get_on_fit_config(cfg.config_fit),
                                         evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader))
    
    ## step 5: start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={'num_cpus': 2, 'num_gpus': 0.125},
    )

    ## step 6: save results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / 'result.pkl'

    results = {'history': history, 'anythingelse': "here"}

    with open(str(results_path), 'wb') as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()