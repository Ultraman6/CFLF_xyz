import torch
from model.cnn import SimpleCNN

class Server:
    def __init__(self, device):
        self.global_model = SimpleCNN().to(device)
        self.device = device

    def aggregate(self, client_states):
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.mean(torch.stack([client_states[i][k].float() for i in range(len(client_states))]), 0)
        self.global_model.load_state_dict(global_dict)

    def get_global_model(self):
        return self.global_model
