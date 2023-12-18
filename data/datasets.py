import random

import numpy as np
from torch.utils.data import Subset, random_split
from torchvision import datasets, transforms

from data import get_data_loader


def load_data(train=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('./data', train=train, download=True, transform=transform)
    return dataset


def load_data_with_validation(train=True, validation_split=0.01):
    dataset = load_data(train)
    if train:
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        return get_data_loader(val_dataset), val_size
    else:
        print("服务器验证集大小为"+str(len(dataset)))
        return dataset, len(dataset)


def split_data(dataset, num_clients):
    total_size = len(dataset)
    indices = list(range(total_size))
    np.random.shuffle(indices)
    split_size = total_size // num_clients # 将数据装载好在分配
    return [(get_data_loader(Subset(dataset, indices[i * split_size:(i + 1) * split_size])), split_size) for i in range(num_clients)]


def split_iid(train_dataset, test_dataset, num_clients):
    train_data_splits = split_data(train_dataset, num_clients)
    test_data_splits = split_data(test_dataset, num_clients)
    client_dataloaders = [(train_data_splits[i], test_data_splits[i]) for i in range(num_clients)]
    return client_dataloaders


def split_non_iid_by_label(dataset, num_clients, imbalance_factor):
    # 确保imbalance_factor在0到1之间
    imbalance_factor = max(0, min(imbalance_factor, 1))

    # 获取每个样本的标签
    labels = np.array(dataset.targets)

    # 计算每个类别的样本数
    num_classes = len(np.unique(labels))
    per_class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    # 计算每个类别的目标样本数
    per_class_counts = [len(indices) for indices in per_class_indices]
    per_class_targets = [int(count * imbalance_factor) for count in per_class_counts]

    # 为每个客户端分配样本
    clients_data = [[] for _ in range(num_clients)]
    for class_indices, target in zip(per_class_indices, per_class_targets):
        np.random.shuffle(class_indices)
        split_indices = np.array_split(class_indices, num_clients)
        for i, client_indices in enumerate(split_indices):
            if len(clients_data[i]) < target:
                clients_data[i].extend(client_indices)

    # 创建Subset
    clients_data = [Subset(dataset, indices) for indices in clients_data]

    return clients_data


def split_non_iid_by_sample(dataset, num_clients, imbalance_factor):
    total_size = len(dataset)
    indices = list(range(total_size))
    np.random.shuffle(indices)

    # 计算每个客户端的数据量
    min_size = int(total_size / num_clients * imbalance_factor)
    sizes = [min_size for _ in range(num_clients)]
    extra = total_size - min_size * num_clients
    for i in np.random.choice(range(num_clients), extra, replace=False):
        sizes[i] += 1

    # 分配数据给客户端
    clients_data = []
    idx = 0
    for size in sizes:
        client_indices = indices[idx: idx + size]
        clients_data.append(Subset(dataset, client_indices))
        idx += size

    return clients_data
