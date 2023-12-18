import torch
from torch.utils.data import DataLoader

from data import load_data
from model.CNN.cnn import SimpleCNN
import torch.nn.functional as F


class Server:
    def __init__(self, device, valid_dataset, valid_sample_number):
        self.global_model = SimpleCNN().to(device)
        self.device = device
        self.valid_dataset = valid_dataset
        self.valid_sample_number = valid_sample_number
        print("服务器初始化完成，验证数据量为{}".format(self.valid_sample_number))

    def aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num
        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            # print("______k = " + str(k))
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        self.global_model.load_state_dict(averaged_params)

    def get_global_model(self):
        return self.global_model

    def valid_global_model(self):
        self.global_model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.valid_dataset:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.valid_dataset)
        accuracy = 100. * correct / self.valid_sample_number
        return test_loss, accuracy
