import torch
import torch.optim as optim
from model.cnn import SimpleCNN
from data.dataloader import get_data_loader

class Client:
    def __init__(self, dataset, device, local_epochs):
        self.model = SimpleCNN().to(device)
        self.train_loader = get_data_loader(dataset)
        self.device = device
        self.local_epochs = local_epochs

    def train(self, global_model):
        self.model.load_state_dict(global_model.state_dict())
        self.model.to(self.device)  # 确保模型在GPU上
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
        for _ in range(self.local_epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = torch.nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()
        return self.model.state_dict()
