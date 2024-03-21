
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
    

if __name__ == "__main__":
    main()

