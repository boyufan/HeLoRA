from peft import LoraConfig, get_peft_model, TaskType
import torch
import copy

from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, DistilBertForSequenceClassification

from peft import LoraConfig, TaskType
from transformers.modeling_outputs import SequenceClassifierOutput


CHECKPOINT = "distilbert-base-uncased"  # transformer model checkpoint
# MODEL = AutoModelForSequenceClassification.from_pretrained(
#         CHECKPOINT, 
#         num_labels=2
#     )



class CustomDistilBertForSequenceClassification(DistilBertForSequenceClassification):
    def forward(self, input_ids: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None, head_mask: torch.Tensor | None = None, inputs_embeds: torch.Tensor | None = None, labels: torch.LongTensor | None = None, output_attentions: bool | None = None, output_hidden_states: bool | None = None, return_dict: bool | None = None) -> SequenceClassifierOutput | torch.Tuple[torch.Tensor]:
        return super().forward(input_ids, attention_mask, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)

# MODEL = DistilBertForSequenceClassification.from_pretrained(
#     CHECKPOINT,
#     num_labels=2
# )

MODEL = CustomDistilBertForSequenceClassification.from_pretrained(
    CHECKPOINT,
    num_labels=2
)


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


if __name__ == "__main__":
    model = build_lora_model(MODEL)
    # model = MODEL
    print(model)