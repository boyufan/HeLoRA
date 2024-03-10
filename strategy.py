from typing import List, Tuple, Union, Dict

from flwr.common import EvaluateIns, EvaluateRes, FitIns, Metrics, Parameters, parameters_to_ndarrays
from flwr.common.typing import FitRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import flwr as fl

from model import get_parameters


# class HeteroLoRA(fl.server.strategy.Strategy):
#     def __init__(self, net, fraction_fit, min_fit_clients, min_available_clients):
#         super().__init__()
#         self.fraction_fit = fraction_fit
#         self.net = net
#         self.min_fit_clients = min_fit_clients
#         self.min_available_clients = min_available_clients
    
#     def __repr__(self) -> str:
#         return "HeterLoRA"
    
#     def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
#         ndarrays = get_parameters(self.net)
#         return fl.common.ndarrays_to_parameters(ndarrays)
    
    
#     def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
#         sample_size, min_num_clients = self.num_fit_clients(
#             client_manager.num_available()
#         )
#         clients = client_manager.sample(
#             num_clients=sample_size, min_num_clients=min_num_clients
#         )
#         print(f"in configure fit , server round no. = {server_round}")
    
#     def aggregate_fit(self, server_round, result, failures):
#         print("aggregate fit")
#         print(f"the return results are {result}")

#     def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
#         print("configure evaluate with nothing")

#     def evaluate(self, server_round: int, parameters: Parameters):
#         print("no evaluate at the moment")

#     def aggregate_evaluate(self, server_round, results, failures):
#         print("no evaluate at the moment")
    

#     def _zero_padding(self):
#         '''A implementation of zero_padding strategy for heterogeneous LoRA'''
#         print("zero_padding")
    
#     def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
#         """Return sample size and required number of clients."""
#         num_clients = int(num_available_clients * self.fraction_fit)
#         return max(num_clients, self.min_fit_clients), self.min_available_clients

#     def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
#         """Use a fraction of available clients for evaluation."""
#         num_clients = int(num_available_clients * self.fraction_evaluate)
#         return max(num_clients, self.min_evaluate_clients), self.min_available_clients


#TODO: change aggregate strategy: 1) obtain the heterogeneous model parameters 2) padding-zero 3) aggregate
class HeteroLoRA(fl.server.strategy.FedAvg):
    def __init__(self, net, fraction_fit, min_fit_clients, min_available_clients, initial_parameters):
        super().__init__()
        self.net = net
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters


    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        ndarrays = get_parameters(self.net)
        return fl.common.ndarrays_to_parameters(ndarrays)


    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy | FitRes]], failures: List[Tuple[ClientProxy, FitRes] | BaseException]) -> Tuple[Parameters | Dict[str, bool | bytes | float | int | str] | None]:
        # if not results:
        #     return None, {}
        # if not self.accept_failures and failures:
        #     return None, {}
        
        # print("aggregate the parameters")

        # aggregate_parameters = []

        # for _client, res in results:
        #     params = parameters_to_ndarrays(res.parameters)
        #     aggregate_parameters.append(params)

        # agg_cum_gradient = aggregate()

        return super().aggregate_fit(server_round, results, failures)



    

        
        
