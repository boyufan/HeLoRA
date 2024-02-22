import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST

import numpy as np

from datasets import load_dataset

from transformers import AutoTokenizer

def get_mnist(data_path='./data'):

    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)

    return trainset, testset



def prepare_dataset(num_partitions, batch_size, val_ratio=0.1):

    trainset, testset = get_mnist()

    # split trainset into 'num_partitions' trainsets
    num_images = len(trainset) // num_partitions
    partition_len = [num_images] * num_partitions
    trainset = random_split(trainset, partition_len, torch.Generator().manual_seed(2024))

    # creat dataloaders with train+val support
    trainloaders = []
    valloaders = []
    for trainset_ in trainset:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2024))

        trainloaders.append(DataLoader(for_train, batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))
    
    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader



def load_huggingface_dataset(dataset_name: str, num_clients: int, iid=True, alpha=1.0):

    train_dataset = load_dataset("imdb", split='train')
    test_dataset = load_dataset("imdb", split='test')

    split_dataset = train_dataset.train_test_split(test_size=0.1, seed=2024)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

    client_traindata = []

    if iid:
        total_size = len(tokenized_train_dataset)
        per_client_size = total_size // num_clients
        indices = np.random.permutation(total_size)
        for i in range(num_clients):
            start_idx = i * per_client_size
            end_idx = (i + 1) * per_client_size if i < num_clients - 1 else total_size
            client_indices = indices[start_idx:end_idx]
            client_dataset = tokenized_train_dataset.select(client_indices)
            client_traindata.append(client_dataset)

    # TODO: non-IID
    
    print(client_traindata)
    return client_traindata, tokenized_test_dataset



if __name__ == "__main__":
    load_huggingface_dataset("imdb", 10)
















