from collections import OrderedDict
from typing import List, Tuple, Union, Dict

from flwr.common import EvaluateIns, EvaluateRes, FitIns, Metrics, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.typing import FitRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import flwr as fl

import torch

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
    def __init__(self, net, fraction_fit, min_fit_clients, min_available_clients, initial_parameters, r_values):
        super().__init__()
        self.net = net
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters
        self.r_values = r_values


    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        ndarrays = get_parameters(self.net)
        # 返回序列化的模型参数（bytes）
        return fl.common.ndarrays_to_parameters(ndarrays)


    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy | FitRes]], failures: List[Tuple[ClientProxy, FitRes] | BaseException]) -> Tuple[Parameters | Dict[str, bool | bytes | float | int | str] | None]:
        # if not results:
        #     return None, {}
        # if not self.accept_failures and failures:
        #     return None, {}
        
        print("start aggregating the parameters")

        print(f"the shape of result is {len(results)}")

        # parameters_in_ndarrays = parameters_to_ndarrays(results[0][1].parameters)

        # 这里从fit_res拿到的parameters，类型是Parameters，里面有tensor，tensor里是装着bytes的list
        #TODO: fix the bug here!
        parameters = [fit_res.parameters for _, fit_res in results]
        parameters_in_ndarrays = [parameters_to_ndarrays(parameter) for parameter in parameters]
        padded_parameters = self._zero_padding(parameters_in_ndarrays, self.r_values)
        padded_parameters = ndarrays_to_parameters(padded_parameters)

        # updated_results = [(result[0], result[1]._replace(parameters=padded_param)) for result, padded_param in zip(results, padded_parameters)]

        updated_results = []

        for result, padded_param in zip(results, padded_parameters):
            updated_result = (result[0], result[1])
            updated_result[1].parameters = padded_param
            updated_results.append(updated_result)

        # return super().aggregate_fit(server_round, results, failures)
        return super().aggregate_fit(server_round, updated_results, failures)
    
    def _zero_padding(self, parameters, r_values, max_r=8):
        '''Perform zero_padding for models with smaller r than the global one'''

        # check the correctness!

        padded_parameters = []

        for parameter, r in zip(parameters, r_values):
            params_dict = zip(self.net.state_dict().keys(), parameter)
            state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})

            adapted_state_dict = OrderedDict()

            for key, tensor in state_dict.items():
                if "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                    padding_size = max_r - r
                    if padding_size > 0:
                        if "lora_A.default.weight" in key:
                            padded_tensor = torch.cat([tensor, torch.zeros(padding_size, tensor.size(1))], dim=0)
                        elif "lora_B.default.weight" in key:
                            padded_tensor = torch.cat([tensor, torch.zeros(tensor.size(0), padding_size)], dim=1)
                    else:
                        padded_tensor = tensor
                    adapted_state_dict[key] = padded_tensor
                else:
                    adapted_state_dict[key] = tensor
            print(f"the parameter after padding: {adapted_state_dict}")
            padded_parameter = list(adapted_state_dict.values()) 
            padded_parameters.append(padded_parameter)
            
        
        return padded_parameters




    

        
        
