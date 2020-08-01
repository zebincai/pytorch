from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loader(cfg, phase="train"):
    if phase == "train":
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        data_set = datasets.MNIST(root=cfg.DATA.train_path,
                                  train=True,
                                  download=True,
                                  transform=transformer)
        data_loader = DataLoader(dataset=data_set,
                                 batch_size=cfg.TRAIN.batch_size,
                                 shuffle=True,
                                 num_workers=cfg.TRAIN.n_workers,
                                 pin_memory=True)
        return data_loader

    elif phase == "valid":
        pass
    elif phase == "test":
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        data_set = datasets.MNIST(root=cfg.DATA.test_path,
                                  train=False,
                                  download=True,
                                  transform=transformer)

        data_loader = DataLoader(dataset=data_set,
                                 batch_size=cfg.TRAIN.batch_size,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True)
        return data_loader
