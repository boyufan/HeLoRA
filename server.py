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


def get_evaluate_fn(net, num_classes: int, testloader):

    def evaluate_fn(server_round, parameters, config):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = net

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(model, testloader, device)

        return loss, {"accuracy": accuracy}

    
    return evaluate_fn


def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

    
    
