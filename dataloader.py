import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        args = torch.tensor(self.dataframe['encoded_arg'])
        key_points = torch.tensor(self.dataframe['encoded_kp'])
        labels = torch.tensor(self.dataframe['label'])

        return args, key_points, labels


class CustomDataLoader:
    def __init__(self, dataframes, batch_size):
        self.train_data = CustomDataset(dataframes['train'])
        self.eval_data = CustomDataset(dataframes['dev'])
        self.test_data = CustomDataset(dataframes['test'])
        self.batch_size = batch_size

    def get_data_loaders(self):
        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True)
        eval_loader = DataLoader(
            self.eval_data, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False)

        return train_loader, eval_loader, test_loader
