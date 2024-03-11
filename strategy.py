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
        
        print("start aggregating the parameters")
        # 从fit_res拿到的parameters，类型是Parameters，里面有tensor，tensor里是装着bytes的list

        # 这里results的顺序是随机的，并不按照顺序
        # 添加逻辑：只有r小的模型参数才padding

        # a demo, to make sure the order is consistent (temporary naive solution)
        parameters = [fit_res.parameters for _, fit_res in results]
        if results[0][0].cid != "0":
            parameters[0], parameters[1] = parameters[1], parameters[0]
        
        parameters_in_ndarrays = [parameters_to_ndarrays(parameter) for parameter in parameters]
        padded_parameters = self._zero_padding(parameters_in_ndarrays, self.r_values)
        padded_parameters_in_ndarrays = [ndarrays_to_parameters(padded_parameter) for padded_parameter in padded_parameters]

        updated_results = []

        for result, padded_param in zip(results, padded_parameters_in_ndarrays):
            updated_result = (result[0], result[1])
            updated_result[1].parameters = padded_param
            updated_results.append(updated_result)

        # return super().aggregate_fit(server_round, results, failures)
        return super().aggregate_fit(server_round, updated_results, failures)
    
    def _zero_padding(self, parameters, r_values, max_r=8):
        '''Perform zero_padding for models with smaller r than the global one'''
        # 这里的逻辑是通过外部的r_values来判断哪个模型需要padding，但存在的问题是传入的parameters的顺序是随机的，并不一定和r_values的顺序一一对应

        # check the correctness!
        padded_parameters = []

        for parameter, r in zip(parameters, r_values):
            params_dict = zip(self.net.state_dict().keys(), parameter)
            state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})

            adapted_state_dict = OrderedDict()

            for key, tensor in state_dict.items():
                if "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                    print(f"current r is {r}")
                    padding_size = max_r - r
                    print(f"find lora, the padding size is {padding_size}")
                    if padding_size > 0:
                        if "lora_A.default.weight" in key:
                            print(f'the current dimension of loraA before padding is {tensor.shape}')
                            padded_tensor = torch.cat([tensor, torch.zeros(padding_size, tensor.size(1))], dim=0)
                            print(f'the current dimension of loraA is {padded_tensor.shape}')
                        elif "lora_B.default.weight" in key:
                            print(f'the current dimension of loraB before padding is {tensor.shape}')
                            padded_tensor = torch.cat([tensor, torch.zeros(tensor.size(0), padding_size)], dim=1)
                            print(f'the current dimension of loraB is {padded_tensor.shape}')
                    else:
                        padded_tensor = tensor
                    adapted_state_dict[key] = padded_tensor
                else:
                    adapted_state_dict[key] = tensor
                    
            padded_parameter = [tensor.cpu().numpy() for tensor in adapted_state_dict.values()]
            padded_parameters.append(padded_parameter)
            
        
        return padded_parameters




    

        
        
