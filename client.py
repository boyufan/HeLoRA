from collections import OrderedDict
import flwr as fl
from flwr_datasets import FederatedDataset

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from evaluate import load as load_metric

from lora import build_lora_model, build_hetero_lora_models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def train(net, trainloader, epochs):
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    net.to(DEVICE)
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def test(net, testloader):
    metric = load_metric("accuracy")
    loss = 0
    net.eval()
    net.to(DEVICE)
    for batch in testloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, trainloader, r, num_class, hetero, apply_kd) -> None:
        super().__init__()

        self.model = model
        self.cid = cid
        self.trainloader = trainloader
        self.r = r
        self.hetero = hetero
        # self.valloader = valloader
        # self.model = Net(num_class)
        self.apply_kd = apply_kd
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(self.model)
        print(f"the current client is {self.cid}")
    

    def set_parameters(self, parameters):

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    

    def get_parameters(self, config):
        
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    

    def fit(self, parameters, config):

        # copy parameters sent by the server into client's local model
        # here to add extra truncate
        print(f"the shape of parameters: {len(parameters)}")
        
        if self.hetero and not self.apply_kd:
            print(f"truncate the parameters")
            parameters = self._truncate_model(parameters, self.cid, self.r)
        # 对于KD，在set_parameters这里要作单独的对应处理！
        # 现在的问题是只能从server段收到一个全局模型，没有办法收到多个模型参数，这个是现在的瓶颈
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=1)
        print("local train finished!")
        return self.get_parameters({}), len(self.trainloader), {}
    

    def _truncate_model(self, parameters, cid, r):
        """Truncate the global model to fit the local model's parameter"""

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        adapted_state_dict = state_dict.copy()
        
        for key, tensor in adapted_state_dict.items():
            if "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                if "lora_A" in key:
                    adjusted_tensor = tensor[:r, :]
                else:
                    adjusted_tensor = tensor[:, :r]
                adapted_state_dict[key] = adjusted_tensor
            else:
                adapted_state_dict[key] = tensor

        # change back to parameter
        new_parameters = [v for k, v in adapted_state_dict.items()]

        return new_parameters
    


class FlowerClientKD(fl.client.NumPyClient):
    def __init__(self, model, cid, trainloader, testloader) -> None:
        super().__init__()
        self.model = model
        self.cid = cid
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"the current client is {self.cid}")
    

    def set_parameters(self, parameters):
        index = int(self.cid) - 1
        parameter = parameters[index]
        params_dict = zip(self.model.state_dict().keys(), parameter)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    

    def get_parameters(self, config):
        
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    

    def fit(self, parameters, config):

        # copy parameters sent by the server into client's local model
        # for the first round, should be the same with the defination
        print(f"the shape of parameters: {len(parameters)}")
        print(f"the current server round is ")
        print(config["current_round"])
        
        # 对于KD，在set_parameters这里要作单独的对应处理！
        # 现在的问题是只能从server段收到一个全局模型，没有办法收到多个模型参数，这个是现在的瓶颈
        if config["current_round"] != 1:
            self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=1)
        print("local train finished!")
        return self.get_parameters({}), len(self.trainloader), {}

    

    def evaluate(self, parameters, config):
        index = int(self.cid) - 1
        parameter = parameters[index]
        self.set_parameters(parameter)
        # loss, accuracy = test(self.model, self.valloader, self.device)
        loss, accuracy = test(self.model, self.valloader)
        # print(f'loss: {loss}, accuracy: {accuracy}')
        return float(loss), len(self.valloader), {'accuracy': accuracy}
    

class HuaggingFaceClient(fl.client.NumPyClient):
    def __init__(self, model, tokenizer, train_dataset, testdataset) -> None:
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset

        self.training_args = TrainingArguments(output_dir="./huggingface_output", evaluation_strategy="epoch")

    def get_parameters(self):

        return super().get_parameters([val.cpu().numpy() for _, val in self.model.state_dict().items()])

    def set_parameters(self, parameters):

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):

        self.set_parameters(parameters)
        trainer = Trainer(
            model = self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
        )
        trainer.train()
        return self.get_parameters(), len(self.train_dataset), {}
    
    def evaluate(self, parameters, config):

        self.set_parameters(parameters)
        return 0.0, len(self.train_dataset), {"accuracy": 0.0}


def generate_client_fn(trainloaders, num_classes, CHECKPOINT, r, hetero: bool = False, apply_kd: bool = False):

    net = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, 
        num_labels=num_classes
    )

    if not hetero:
        print("homogeneous setting")
        # lora_net = build_lora_model(net)

    else:
        print("heterogeneous setting")
        # lora_nets = build_hetero_lora_models(net, r)
    lora_net = build_lora_model(net)
    lora_nets = build_hetero_lora_models(net, r)

    def client_fn(cid: str):
        if not hetero:
            return FlowerClient(model=lora_net,
                                cid=cid,
                                trainloader=trainloaders[int(cid)],
                                r=r,
                                num_class=num_classes,
                                hetero=hetero,
                                apply_kd=apply_kd).to_client()
        else:
            return FlowerClient(model=lora_nets[int(cid)],
                                cid=cid,
                                trainloader=trainloaders[int(cid)],
                                r=r[int(cid)],
                                num_class=num_classes,
                                hetero=hetero,
                                apply_kd=apply_kd).to_client()
    return client_fn, lora_net, lora_nets



def generate_client_fn_kd(trainloaders, testloader, num_classes, CHECKPOINT, r):

    net = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, 
        num_labels=num_classes
    )

    lora_nets = build_hetero_lora_models(net, r)

    def client_fn(cid: str):

        return FlowerClientKD(model=lora_nets[int(cid)],
                              cid=cid,
                              trainloader=trainloaders[int(cid)],
                              testloader=testloader,
                              )

    return client_fn, lora_nets


def main():
    print("test")

if __name__ == "__main__":
    main()

    
