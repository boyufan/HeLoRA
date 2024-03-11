
from peft import LoraConfig, get_peft_model, TaskType
import torch
import copy

from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification

from peft import LoraConfig, TaskType


CHECKPOINT = "distilbert-base-uncased"  # transformer model checkpoint
MODEL = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, 
        num_labels=2
    )


def build_lora_model(model):
    
    # print(f"model without lora:\n {model}")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_lin", "k_lin"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    lora_model = get_peft_model(model, lora_config)
    # print(f"model after lora:\n {lora_model}")
    # lora_model.print_trainable_parameters()
    return lora_model


def build_hetero_lora_models(model, r_values):

    lora_models = []

    for value in r_values:
        print(f"the current r is: {value}")
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
        # print(f"model after lora:\n {lora_model}")
        lora_models.append(lora_model)
    # lora_model.print_trainable_parameters()
        
    # print(f"the heterogeneous models are: {lora_models}")
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

if __name__ == "__main__":
    lora = build_lora_model(MODEL)