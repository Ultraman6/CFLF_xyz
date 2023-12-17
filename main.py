import torch
import matplotlib.pyplot as plt
from algo.fedavg.fedavg_api import run_federated_learning
from data.datasets import load_data
from torch.utils.data import DataLoader
import torch.nn.functional as F

# ... test_model 函数保持不变 ...

def main():
    num_clients = 10
    global_epochs = 2  # 全局轮次
    local_epochs = 5    # 本地轮次
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Algorithm: Federated Learning")
    print("Dataset: MNIST")
    print("Model: Simple CNN")

    global_model, accuracy_list, loss_list = run_federated_learning(num_clients, global_epochs, local_epochs, device)

    # 绘制精度和损失图表
    epochs = range(1, global_epochs + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy_list, 'o-', label='Accuracy')
    plt.title('Accuracy over Global Epochs')
    plt.xlabel('Global Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss_list, 'o-', label='Loss')
    plt.title('Loss over Global Epochs')
    plt.xlabel('Global Epochs')
    plt.ylabel('Loss')

    plt.show()

if __name__ == "__main__":
    main()
