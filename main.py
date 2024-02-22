
import hydra
import pickle
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import flwr as fl
from dataset import prepare_dataset
from client import generate_client_fn, load_data
from server import get_evaluate_fn, get_on_fit_config


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    ## step 1: load dataset
    trainloaders, valloaders, _ = load_data(cfg.num_clients)

    ## step 2 define client
    client_fn = generate_client_fn(trainloaders, valloaders, cfg.num_classes)

    ## step 3 define strategy
    strategy = fl.server.strategy.FedAvg(fraction_fit=1.0,
                                         fraction_evaluate=0.5,
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