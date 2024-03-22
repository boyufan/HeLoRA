
from peft import LoraConfig, get_peft_model, TaskType
import torch
import copy

from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from peft import LoraConfig, TaskType
from dataset import load_public_data, load_federated_data
from torch.optim import AdamW

from datasets import load_dataset
from torch.utils.data import random_split, DataLoader


CHECKPOINT = "distilbert-base-uncased"  # transformer model checkpoint
MODEL = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, 
        num_labels=2
    )
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def build_lora_model(model):
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_lin", "k_lin"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    lora_model = get_peft_model(model, lora_config)
    return lora_model


def get_dataloader(dataname):
    dataset = load_dataset(dataname, split="train")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns("text")
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=32, collate_fn=data_collator)

    return dataloader


def train(model, dataloader):
    model.to(DEVICE)
    model.train()

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(1):
        for batch in dataloader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            print(outputs)
            loss = outputs[0]
            logits = outputs[1]
            loss.backward()
            optim.step()
            break


def main():
    model = build_lora_model(MODEL)
    print(model)
    # dataloader = get_dataloader("imdb")
    # train(model, dataloader)






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
        # assume we got the parameters list from the aggregate_fit(), which consists of model parameters for heterogeneous models
        # then we need to use cid to choose the corresponding one
        index = int(self.cid) - 1
        parameter = parameters[index]

        params_dict = zip(self.model.state_dict().keys(), parameter)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    

    def get_parameters(self, config):
        
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    

    def fit(self, parameters, config):
        
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=1)
        print("local train finished!")
        return self.get_parameters({}), len(self.trainloader), {}
    


# def aggregate_fit(self, server_round, results, failures):
    
#     results = sorted(results, key=lambda x: int(x[0].cid))
#     parameters = [fit_res.parameters for _, fit_res in results]
#     parameters_in_ndarrays = [parameters_to_ndarrays(parameter) for parameter in parameters]
#     kd_parameters = self._kd_aggregate(parameters_in_ndarrays, self.hetero_net)
#     kd_parameters = ndarrays_to_parameters(kd_parameters)
#     return kd_parameters








    

if __name__ == "__main__":
    main()

