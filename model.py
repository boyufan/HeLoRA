from typing import List, OrderedDict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from flwr.common import parameters_to_ndarrays

from transformers import AutoModelForSequenceClassification, DistilBertForSequenceClassification, DistilBertTokenizer

from evaluate import load as load_metric
from transformers.modeling_outputs import SequenceClassifierOutput
from lora import build_lora_model

import numpy as np


# hardcode the model at the moment

CHECKPOINT = "distilbert-base-uncased"
Net = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, 
        num_labels=2
    )
Net = build_lora_model(Net)


class DistilBertForKD(DistilBertForSequenceClassification):
    def forward(self, 
                input_ids=None, 
                attention_mask=None, 
                head_mask=None,
                inputs_embeds=None, 
                labels=None,
                output_attentions=None,
                output_hidden_states=None, 
                return_dict=None):
        

        
        return super().forward(input_ids, attention_mask, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)



def get_parameters(net) -> List[np.ndarray]:
    """Return the parameters of model as numpy.NDArrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    """Set the model parameters with given parameters."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_state_dict_from_param(model, parameters):
    """Get the state dict from model & parameters as np.NDarrays.

    Parameters
    ----------
    model : nn.Module
        The neural network.
    parameters : np.NDarray
        Parameters of the model as np.NDarrays.

    Returns
    -------
    Dict
        state dict of model.
    """
    # Load the parameters into the model
    for param_tensor, param_ndarray in zip(
        model.state_dict(), parameters_to_ndarrays(parameters)
    ):
        model.state_dict()[param_tensor].copy_(torch.from_numpy(param_ndarray))
    # Step 3: Obtain the state_dict of the model
    state_dict = model.state_dict()
    return state_dict

# class Net(nn.Module):
#     def __init__(self, num_classes) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 4 * 4)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x 


# def train(net, trainloader, optimizer, epochs, device):
#     """Train the network on the training set."""
#     criterion = torch.nn.CrossEntropyLoss()
    
#     net.train()
#     net.to(device)
#     for epoch in range(epochs):
        
#         for images, labels in trainloader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = net(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             # Metrics
#         #     epoch_loss += loss
#         #     total += labels.size(0)
#         #     correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
#         # epoch_loss /= len(trainloader.dataset)
#         # epoch_acc = correct / total
#         # if verbose:
#         #     print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")




# def test(net, testloader, device):
#     """Evaluate the network on the entire test set."""
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, total, loss = 0, 0, 0.0
#     net.eval()
#     net.to(device)
#     with torch.no_grad():
#         for images, labels in testloader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = net(images)
#             loss += criterion(outputs, labels).item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     loss /= len(testloader.dataset)
#     accuracy = correct / total
#     return loss, accuracy


def test(net, testloader, device):
    metric = load_metric("accuracy")
    loss = 0
    net.eval()
    net.to(device)
    for batch in testloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy


