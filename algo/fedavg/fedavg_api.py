# ... 其他代码保持不变 ...
import concurrent.futures
import copy

import torch
from tqdm import tqdm

from algo.fedavg.client import Client
from algo.fedavg.server import Server
from data import load_data, split_data
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data.datasets import split_iid, split_non_iid_by_label, load_data_with_validation


def train_client(client, global_model):
    return [client.local_train_num, client.train(global_model)]


def run_federated_learning(num_clients, global_epochs, local_epochs, device):
    # 加载数据集
    train_dataset = load_data(train=True)
    test_dataset = load_data(train=False)
    valid_dataset, valid_sample_number = load_data_with_validation()
    clients_dataset = split_iid(train_dataset, test_dataset, num_clients)

    server = Server(device, valid_dataset, valid_sample_number)
    w_global = server.get_global_model()
    clients = [Client(idx, client_dataset, device, local_epochs) for idx, client_dataset in
               enumerate(clients_dataset)]

    accuracy_list = []
    loss_list = []

    for i in range(global_epochs):
        client_models=[]

        for client in tqdm(clients, desc=f"Epoch {i + 1}/{global_epochs}", leave=False):
            # 串行训练每个客户端
            client_models.append(client.train(copy.deepcopy(server.get_global_model())))
        server.aggregate(client_models)

        # 验证全局模型并记录精度与损失
        test_loss, test_accuracy = server.valid_global_model()
        accuracy_list.append(test_accuracy)
        loss_list.append(test_loss)
        print("全局轮次" + str(i) + "已经完成")

    return server.get_global_model(), accuracy_list, loss_list
