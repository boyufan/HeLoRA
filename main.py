
import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import prepare_dataset
from client import generate_client_fn


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## Step 1
    print(OmegaConf.to_yaml(cfg))

    ## step 2: dataset
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size)

    ## step 3 define client
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)




if __name__ == "__main__":
    main()