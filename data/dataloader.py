from torch.utils.data import DataLoader


def get_data_loader(dataset, batch_size=8, shuffle=True, num_workers=8):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
