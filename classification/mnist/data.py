from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import cfg


class DataSet(object):
    def __init__(self, data_path="./data"):
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        kwargs = {'num_workers': 1, 'pin_memory': True} if cfg.use_cuda else {}
        self.train_loader = DataLoader(
            dataset=datasets.MNIST(root=data_path,
                                   train=True,
                                   download=True,
                                   transform=transformer),
            batch_size=cfg.TRAIN.batch_size,
            shuffle=True,
            **kwargs)

        self.test_loader = DataLoader(
            dataset=datasets.MNIST(root=data_path,
                                   train=False,
                                   download=True,
                                   transform=transformer),
            batch_size=cfg.TRAIN.batch_size,
            shuffle=False,
            **kwargs)