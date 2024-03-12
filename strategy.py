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


class HeteroLoRA(fl.server.strategy.FedAvg):
    def __init__(self, net, fraction_fit, min_fit_clients, min_available_clients, evaluate_fn, initial_parameters, r_values, hetero, padding_strategy="zero"):
        super().__init__()
        self.net = net
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.initial_parameters = initial_parameters
        self.r_values = r_values
        self.hetero = hetero
        self.padding_strategy = padding_strategy


    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        ndarrays = get_parameters(self.net)
        # 返回序列化的模型参数（bytes）
        return fl.common.ndarrays_to_parameters(ndarrays)


    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy | FitRes]], failures: List[Tuple[ClientProxy, FitRes] | BaseException]) -> Tuple[Parameters | Dict[str, bool | bytes | float | int | str] | None]:
        
        print("start aggregating the parameters")
        if self.hetero:
            results = sorted(results, key=lambda x: int(x[0].cid))
            parameters = [fit_res.parameters for _, fit_res in results]
            
            parameters_in_ndarrays = [parameters_to_ndarrays(parameter) for parameter in parameters]

            if self.padding_strategy == "zero":
                padded_parameters = self._zero_padding(parameters_in_ndarrays, self.r_values)
            elif self.padding_strategy == "mean":
                padded_parameters = self._mean_padding(parameters_in_ndarrays, self.r_values)

            padded_parameters_in_ndarrays = [ndarrays_to_parameters(padded_parameter) for padded_parameter in padded_parameters]

            updated_results = []

            for result, padded_param in zip(results, padded_parameters_in_ndarrays):
                updated_result = (result[0], result[1])
                updated_result[1].parameters = padded_param
                updated_results.append(updated_result)
            
            return super().aggregate_fit(server_round, updated_results, failures)

        return super().aggregate_fit(server_round, results, failures)
    

    def _zero_padding(self, parameters, r_values, max_r=8):
        '''Perform zero_padding for models with smaller r than the global one'''

        padded_parameters = []

        for parameter, r in zip(parameters, r_values):
            params_dict = zip(self.net.state_dict().keys(), parameter)
            state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})

            adapted_state_dict = OrderedDict()

            for key, tensor in state_dict.items():
                if "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                    # print(f"current r is {r}")
                    padding_size = max_r - r
                    # print(f"find lora, the padding size is {padding_size}")
                    if padding_size > 0:
                        if "lora_A.default.weight" in key:
                            # print(f'the current dimension of loraA before padding is {tensor.shape}')
                            padded_tensor = torch.cat([tensor, torch.zeros(padding_size, tensor.size(1))], dim=0)
                            # print(f'the current dimension of loraA is {padded_tensor.shape}')
                        elif "lora_B.default.weight" in key:
                            # print(f'the current dimension of loraB before padding is {tensor.shape}')
                            padded_tensor = torch.cat([tensor, torch.zeros(tensor.size(0), padding_size)], dim=1)
                            # print(f'the current dimension of loraB is {padded_tensor.shape}')
                    else:
                        padded_tensor = tensor
                    adapted_state_dict[key] = padded_tensor
                else:
                    adapted_state_dict[key] = tensor
                    
            padded_parameter = [tensor.cpu().numpy() for tensor in adapted_state_dict.values()]
            padded_parameters.append(padded_parameter)
            
        return padded_parameters
    

    def _mean_padding(self, parameters, r_values, max_r=8):
        '''Perform mean padding for models with smaller r than the global one'''

        padded_parameters = []

        for parameter, r in zip(parameters, r_values):
            params_dict = zip(self.net.state_dict().keys(), parameter)
            state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})

            adapted_state_dict = OrderedDict()

            for key, tensor in state_dict.items():
                if "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                    padding_size = max_r - r
                    if padding_size > 0:
                        mean_value = torch.mean(tensor)
                        if "lora_A.default.weight" in key:
                            padded_tensor = torch.cat([tensor, mean_value.expand(padding_size, tensor.size(1))], dim=0)
                        elif "lora_B.default.weight" in key:
                            padded_tensor = torch.cat([tensor, mean_value.expand(tensor.size(0), padding_size)], dim=1)
                    else:
                        padded_tensor = tensor
                    adapted_state_dict[key] = padded_tensor
                else:
                    adapted_state_dict[key] = tensor
                    
            padded_parameter = [tensor.cpu().numpy() for tensor in adapted_state_dict.values()]
            padded_parameters.append(padded_parameter)
            
        return padded_parameters
    

    def evaluate(self, server_round: int, parameters: Parameters) -> Tuple[float, Dict[str, bool | bytes | float | int | str]] | None:
        return super().evaluate(server_round, parameters)




    

        
        
