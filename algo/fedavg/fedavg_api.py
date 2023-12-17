# ... 其他代码保持不变 ...
import concurrent.futures

import torch

from algo.fedavg.client import Client
from algo.fedavg.server import Server
from data import load_data, split_data
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data.datasets import split_iid


def train_client(client, global_model):
    return client.train(global_model)
def run_federated_learning(num_clients, global_epochs, local_epochs, device):
    dataset = load_data()
    clients_data = split_iid(dataset, num_clients)

    server = Server(device)
    clients = [Client(client_data, device, local_epochs) for client_data in clients_data]

    accuracy_list = []
    loss_list = []

    for i in range(global_epochs):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 使用多线程并行训练客户端
            futures = [executor.submit(train_client, client, server.get_global_model()) for client in clients]
            client_states = [future.result() for future in concurrent.futures.as_completed(futures)]

        server.aggregate(client_states)

        # 测试全局模型并记录精度与损失
        test_loss, test_accuracy = test_model(server.get_global_model(), device)
        accuracy_list.append(test_accuracy)
        loss_list.append(test_loss)
        print("全局轮次" + str(i) + "已经完成")

    return server.get_global_model(), accuracy_list, loss_list

def test_model(model, device):
    test_dataset = load_data(train=False)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy