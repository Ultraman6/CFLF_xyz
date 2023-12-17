import torch
import torch.optim as optim
from model.cnn import SimpleCNN
from data.dataloader import get_data_loader

class Client:
    def __init__(self, train_data, test_data, device):
        self.model = SimpleCNN().to(device)
        self.train_loader = get_data_loader(train_data, batch_size=64, num_workers=4)
        self.test_loader = get_data_loader(test_data, batch_size=64, num_workers=4)
        self.device = device
        self.train_data_size = len(train_data)
        self.test_data_size = len(test_data)

    def train(self, global_model, epochs):
        self.model.load_state_dict(global_model.state_dict())
        self.model.to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)

        self.model.train()
        for epoch in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = torch.nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()

        return self.model.state_dict()

    def evaluate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += torch.nn.CrossEntropyLoss()(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= self.test_data_size
        accuracy = 100. * correct / self.test_data_size
        return test_loss, accuracy

