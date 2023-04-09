
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CifarDataset(Dataset):
    def __init__(self, dataset_path, type="train"):
        super(CifarDataset, self).__init__()
        if type == "train":
            self.dataset = datasets.CIFAR10(dataset_path,
                                            train=True,
                                            download=True,
                                            transform=transforms.ToTensor())
        elif type == "valid":
            self.dataset = datasets.CIFAR10(dataset_path,
                                            train=False,
                                            download=True,
                                            transform=transforms.ToTensor())
        else:
            pass

    def __len__(self):
        self.dataset.data.__len__()

    def __getitem__(self, idx):
        img_tensor = self.dataset.data[idx] # 32 x 32 x 3
        return img_tensor

    def get_dataloader(self, batch_size):
        # If dataset is imbalanced then one could use WeightRandomSampler/others to balance the data while sampling
        dataloader = DataLoader(dataset = self.dataset, batch_size=batch_size)
        return dataloader

if __name__ == "__main__":
    dataset = CifarDataset('./data', "valid")
    dataloader = dataset.get_dataloader(32)
    batch = next(iter(dataloader))
    images, labels = batch
    print(images.shape)
