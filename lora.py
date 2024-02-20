
from peft import LoraConfig, get_peft_model, TaskType
import torch

from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer

from peft import LoraConfig, TaskType

# peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, 
#                          inference_mode=False, 
#                          r=8, 
#                          lora_alpha=32, 
#                          lora_dropout=0.1
#                          )


def build_lora_model(model):
    
    # model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-large")
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