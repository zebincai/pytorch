from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import cfg


class DataSet(object):
    def __init__(self, data_path="./data"):
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        kwargs = {'num_workers': 2, 'pin_memory': True} if cfg.use_cuda else {}
        self.train_loader = DataLoader(
            dataset=datasets.CIFAR10(root=data_path,
                                     train=True,
                                     download=True,
                                     transform=transformer),
            batch_size=cfg.TRAIN.batch_size,
            shuffle=True,
            **kwargs)

        self.test_loader = DataLoader(
            dataset=datasets.CIFAR10(root=data_path,
                                     train=False,
                                     download=True,
                                     transform=transformer),
            batch_size=cfg.TRAIN.batch_size,
            shuffle=False,
            **kwargs)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt
    import numpy as np

    # functions to show an image
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    data_set = DataSet()
    dataiter = iter(data_set.train_loader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % data_set.classes[labels[j]] for j in range(4)))