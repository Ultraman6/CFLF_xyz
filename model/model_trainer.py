import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def train(self, train_loader, epochs, lr=0.01):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix(loss=loss.item())

    def evaluate(self, test_dataset, batch_size=8):
        self.model.eval()
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        return test_loss, accuracy

    def get_model_state(self):
        return self.model.state_dict()

    def set_model_state(self, state_dict):
        self.model.load_state_dict(state_dict)
