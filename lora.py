
from peft import LoraConfig, get_peft_model, TaskType
import torch
import copy

from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification

from peft import LoraConfig, TaskType
from dataset import load_public_data, load_federated_data
from torch.optim import AdamW


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
    # lora_model.print_trainable_parameters()
    return lora_model


def build_hetero_lora_models(model, r_values):

    lora_models = []

    for value in r_values:
        model_copy = copy.deepcopy(model)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            target_modules=["q_lin", "k_lin"],
            inference_mode=False,
            r=value,
            lora_alpha=32,
            lora_dropout=0.1
        )
        lora_model = get_peft_model(model_copy, lora_config)
        lora_models.append(lora_model)
    # lora_model.print_trainable_parameters()
    return lora_models



# training_args = TrainingArguments(
#     output_dir="outputs/bigscience/mt0-large-lora",
#     learning_rate=1e-3,
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=32,
#     num_train_epochs=2,
#     weight_decay=0.01,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )

# trainer.train()

def train(net, trainloader, epochs):
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    net.to(DEVICE)
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            print(f"batch is {batch}")
            outputs = net(**batch)
            print(f"output is {outputs}")
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            break


if __name__ == "__main__":
    lora = build_lora_model(MODEL)
    trainloader, _ = load_federated_data(1, CHECKPOINT)
    train(lora, trainloader[0], 1)

    # print(lora.state_dict())