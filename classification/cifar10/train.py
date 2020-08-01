import torch
import torch.optim as optim
from model import Net
from config import cfg, init_conf
from loss import CELoss
import torch.nn as nn
from data import DataSet
from alglib.model_saver import saver
from pylib.utils.log import init_logger
import logging


class Trainer(object):
    def __init__(self):
        self.device = torch.device("cuda" if cfg.use_cuda else "cpu")
        self.net = Net().to(self.device)
        self.data_set = DataSet()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
                params=self.net.parameters(),
                lr=cfg.TRAIN.lr,
                momentum=cfg.TRAIN.momentum)

    def train_one_epoch(self, epoch):
        # set training model
        self.net.train()
        running_loss = 0.0
        for batch_idx,  (data, target) in enumerate(self.data_set.train_loader):
            # load batch data and set the corresponding device mode
            data = data.to(self.device)
            target = target.to(self.device)

            # zero the gradient
            self.optimizer.zero_grad()
            # infer the data
            output = self.net(data)
            # loss calculation
            loss = self.loss(output, target)
            # gradient  calculation
            loss.backward()
            # back propagation
            self.optimizer.step()
            running_loss += loss.item()
            if batch_idx % cfg.TRAIN.log_interval == 0:
                iter_num = (batch_idx + 1) * len(data)
                logging.info('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, iter_num, len(self.data_set.train_loader.dataset), running_loss/iter_num))

    def eval(self):
        self.net.eval()
        correct = 0
        with torch.no_grad():
            for data, target in self.data_set.test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.net(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        logging.info('\nTest set:Accuracy: {}/{} ({:.02f}%)\n'.format(
            correct, len(self.data_set.test_loader.dataset),
            100. * correct / len(self.data_set.test_loader.dataset)))

    def train(self):
        for epoch in range(1, cfg.TRAIN.epochs + 1):
            self.train_one_epoch(epoch)
            self.eval()
            saver(self.net, cfg.checkpoint, cfg.model_name)


if __name__ == "__main__":
    init_conf(cfg)
    init_logger(file_name=cfg.log_name, directory=cfg.checkpoint)

    trainer = Trainer()
    trainer.train()