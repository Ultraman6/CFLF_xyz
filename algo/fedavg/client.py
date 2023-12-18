import copy

import torch
import torch.optim as optim
from model.CNN.cnn import SimpleCNN
from data.dataloader import get_data_loader


class Client:
    def __init__(self, id, dataset, device, local_epochs):
        self.id = id
        self.model = SimpleCNN()
        self.local_train_data = dataset[0][0]
        self.local_test_data = dataset[1][0]
        self.local_train_num = dataset[0][1]
        self.local_test_num = dataset[1][1]
        self.device = device
        self.local_epochs = local_epochs
        print("客户端{}初始化完成，本地训练数据量为{}".format(self.id, self.local_train_num))

    def update_local_dataset(self, local_training_data, local_test_data, local_sample_number):
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def train(self, global_model):
        self.model = global_model
        self.model.to(self.device)  # 确保模型在GPU上
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
        for _ in range(self.local_epochs):
            for data, target in self.local_train_data:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = torch.nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()
        return self.local_train_num, copy.deepcopy(self.model.state_dict())
