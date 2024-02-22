from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays
import flwr as fl
import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from evaluate import load as load_metric

from flwr_datasets import FederatedDataset

from lora import build_lora_model


# from model import Net, train, test

CHECKPOINT = "distilbert-base-uncased"  # transformer model checkpoint
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

net = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=2
    )

print(net)

lora_net = build_lora_model(net)

print('#' * 80)

print(lora_net)


def load_data(num_clients):
    """Load IMDB data (training and eval)"""
    fds = FederatedDataset(dataset="imdb", partitioners={"train": 5})

    trainloaders = []
    valloaders = []

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    # assign dataloaders to clients
    for i in range(num_clients):
        partition = fds.load_partition(i)

        partition_train_test = partition.train_test_split(test_size=0.2)
        partition_train_test = partition_train_test.map(tokenize_function, batched=True)
        partition_train_test = partition_train_test.remove_columns("text")
        partition_train_test = partition_train_test.rename_column("label", "labels")

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainloader = DataLoader(
            partition_train_test["train"],
            shuffle=True,
            batch_size=32,
            collate_fn=data_collator,
        )

        valloader = DataLoader(
            partition_train_test["test"], batch_size=32, collate_fn=data_collator
        )

        trainloaders.append(trainloader)
        valloaders.append(valloader)
    
    # for the server test
    testset = fds.load_full("test")
    testloader = DataLoader(testset, batch_size=32, collate_fn=data_collator)

    return trainloaders, valloaders, testloader


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
    def __init__(self, model, trainloader, valloader, num_class) -> None:
        super().__init__()

        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        # self.model = Net(num_class)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    def set_parameters(self, parameters):

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)


    def get_parameters(self, config):
        
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    

    def fit(self, parameters, config):

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=1)
        return self.get_parameters({}), len(self.trainloader), {}
    

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)
        # loss, accuracy = test(self.model, self.valloader, self.device)
        loss, accuracy = test(self.model, self.valloader)
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


def generate_client_fn(trainloaders, valloaders, num_classes):

    def client_fn(cid: str):
        return FlowerClient(model=lora_net,
                            trainloader=trainloaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            num_class=num_classes).to_client()
    return client_fn


def main():
    print("test")

if __name__ == "__main__":
    main()

    
