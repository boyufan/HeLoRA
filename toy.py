import hydra
from omegaconf import DictConfig, OmegaConf


def function_test(x, y):
    result = x + y
    print(f'{result = }')
    return result

@hydra.main(config_path="conf", config_name="toy", version_base=None)
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    print(cfg.foo)
    print(cfg.bar.baz)
    print(cfg.bar.more)
    print(cfg.bar.more.blabla)


if __name__ == "__main__":
    main()