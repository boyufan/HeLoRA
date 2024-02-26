from collections import OrderedDict
from typing import List, Tuple
from flwr.common import FitIns, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from omegaconf import DictConfig
import torch

import flwr as fl


from model import Net, test


def get_on_fit_config(config: DictConfig):

    def fig_config_fn(server_round: int):
        
        return {'lr': config.lr, 'momentum': config.momentum, 'local_epochs': config.local_epochs}
    
    return fig_config_fn


def get_evaluate_fn(num_classes: int, testloader):

    def evaluate_fn(server_round, parameters, config):

        model = Net(num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        loss, accuracy = test(model, testloader, device)

        return loss, {"accuracy": accuracy}

    
    return evaluate_fn


class HeteroLora(fl.server.strategy.Strategy):
    def __init__(self) -> None:
        super().__init__()

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy | FitIns]]:


        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, parameters, results, failures):
        return
    
    
