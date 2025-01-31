from collections import OrderedDict
from typing import List, Tuple, Union, Dict

from flwr.common import EvaluateIns, EvaluateRes, FitIns, Metrics, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.typing import FitRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import flwr as fl

import torch
from torch.optim import AdamW

from model import get_parameters
from utilis import MutualEnsemble, KLDiv
from dataset import load_public_data, load_public_data_test

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


class HeLoRAPad(fl.server.strategy.FedAvg):
    def __init__(self, net, fraction_fit, min_fit_clients, min_available_clients, evaluate_fn, initial_parameters, r_values, hetero, hetero_net, padding_strategy="zero"):
        super().__init__()
        self.net = net
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.initial_parameters = initial_parameters
        self.r_values = r_values
        self.hetero = hetero
        self.hetero_net = hetero_net
        self.padding_strategy = padding_strategy


    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        ndarrays = get_parameters(self.net)
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
            elif self.padding_strategy == "linear":
                padded_parameters = self._linear_padding(parameters_in_ndarrays, self.r_values) 
            elif self.padding_strategy == "kd":
                kd_parameters = self._kd_aggregate(parameters_in_ndarrays, self.hetero_net)
                
                # kd_parameters_in_ndarrays = [ndarrays_to_parameters(kd_parameter) for kd_parameter in kd_parameters]
                # return kd_parameters_in_ndarrays, {}

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
    

    def _linear_padding(self, parameters, r_values, max_r=8):
        '''Perform linear padding for models with smaller r than the global one'''

        padded_parameters = []

        for parameter, r in zip(parameters, r_values):
            params_dict = zip(self.net.state_dict().keys(), parameter)
            state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})

            adapted_state_dict = OrderedDict()

            for key, tensor in state_dict.items():
                padding_size = max_r - r
                if "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                    if padding_size > 0:
                        # Determine padding for lora_A and lora_B differently based on their dimensions
                        if "lora_A.default.weight" in key:
                            # Padding on the last dimension for lora_A
                            values_to_interpolate = tensor.mean(dim=0, keepdim=True).expand(padding_size, -1)
                            padded_tensor = torch.cat([tensor, values_to_interpolate], dim=0)
                        elif "lora_B.default.weight" in key:
                            # Padding on the last dimension for lora_B
                            values_to_interpolate = tensor.mean(dim=1, keepdim=True).expand(-1, padding_size)
                            padded_tensor = torch.cat([tensor, values_to_interpolate], dim=1)
                    else:
                        padded_tensor = tensor
                    adapted_state_dict[key] = padded_tensor
                else:
                    adapted_state_dict[key] = tensor
                    
            padded_parameter = [tensor.cpu().numpy() for tensor in adapted_state_dict.values()]
            padded_parameters.append(padded_parameter)
            
        return padded_parameters
    

    def _adjacent_padding(self, parameters, r_values, max_r=8):
        '''Perform adjacent padding for models with smaller r than the global one by copying the last row/column'''

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
                            # Copy last row for lora_A
                            last_row = tensor[-1:, :]  # Get last row
                            padding = last_row.repeat(padding_size, 1)
                            padded_tensor = torch.cat([tensor, padding], dim=0)
                        elif "lora_B.default.weight" in key:
                            # Copy last column for lora_B
                            last_col = tensor[:, -1:]  # Get last column
                            padding = last_col.repeat(1, padding_size)
                            padded_tensor = torch.cat([tensor, padding], dim=1)
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



class HeLoraKD(fl.server.strategy.FedAvg):
    def __init__(self, net, fraction_fit, min_fit_clients, min_available_clients, r_values, hetero_net, on_fit_config_fn):
        super().__init__()
        self.net = net
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.r_values = r_values
        self.hetero_net = hetero_net
        self.on_fit_config_fn = on_fit_config_fn


    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy | FitRes]], failures: List[Tuple[ClientProxy, FitRes] | BaseException]) -> Tuple[Parameters | Dict[str, bool | bytes | float | int | str] | None]:
        
        print("start aggregating the parameters")
        results = sorted(results, key=lambda x: int(x[0].cid))
        parameters = [fit_res.parameters for _, fit_res in results]
        # parameters_in_ndarrays = parameters_to_ndarrays(parameters)
        parameters_in_ndarrays = [parameters_to_ndarrays(parameter) for parameter in parameters]
        kd_parameters = self._kd_aggregate(parameters_in_ndarrays, self.hetero_net)
        
        kd_parameters = ndarrays_to_parameters(kd_parameters)
        # kd_parameters_in_ndarrays = [ndarrays_to_parameters(kd_parameter) for kd_parameter in kd_parameters]
        return kd_parameters

    
    def _kd_aggregate(self, parameters, hetero_nets):
        current_net = []
        optimizers = []

        for parameter, net in zip(parameters, hetero_nets):

            params_dict = zip(net.state_dict().keys(), parameter)
            state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})
            net.load_state_dict(state_dict)
            net.to(DEVICE)
            net.train()

            current_net.append(net)
        
        print(f"load finished, the number is {len(current_net)}") 

        # calculate the average logit
        ensemble_model = MutualEnsemble(current_net)

        # public_dataloader = load_public_data("imdb")
        public_dataloader = load_public_data_test()

        criterion = KLDiv(T=1)

        optimizer_1 = AdamW(current_net[0].parameters(), lr=5e-5)
        optimizer_2 = AdamW(current_net[1].parameters(), lr=5e-5)
        optimizer_3 = AdamW(current_net[2].parameters(), lr=5e-5)
        optimizers.append(optimizer_1)
        optimizers.append(optimizer_2)
        optimizers.append(optimizer_3)


        for epoch in range(10):
            for batch in public_dataloader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                with torch.no_grad():
                    average_logit = ensemble_model(**batch)
                
                for index, model in enumerate(current_net):
                    model.zero_grad()
                    batch = {k: v.to(DEVICE) for k, v in batch.items()}
                    logit = model(**batch).logits
                    loss_kd = criterion(logit, average_logit)
                    loss_kd.backward()
                    optimizers[index].step()
        
        # return the updated model parameters
        return_parameters = []

        for index, model in enumerate(current_net):
            state_dict_model = model.state_dict()
            parameter = [tensor.cpu().numpy() for tensor in state_dict_model.values()]
            return_parameters.append(parameter)

        return return_parameters



    def evaluate(self, server_round: int, parameters: Parameters) -> Tuple[float, Dict[str, bool | bytes | float | int | str]] | None:
        return super().evaluate(server_round, parameters)

        
        
