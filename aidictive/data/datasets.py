
import torch
from torchvision import datasets, transforms


def mnist_datasets(directory, num_output_channels=1):
    ds_train = datasets.MNIST(directory, train=True, download=True,
        transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=num_output_channels),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]))
    ds_test = datasets.MNIST(directory, train=False,
        transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=num_output_channels),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]))
    return ds_train, ds_test


def mnist_data_loaders(directory, batch_size=64, batch_size_test=1024,
                       **kwargs):
    ds_train, ds_test = mnist_datasets(directory, **kwargs)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size,
                                           shuffle=True)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size_test,
                                          shuffle=False)
    return dl_train, dl_test


# Define some function aliases.
mnist = mnist_datasets

