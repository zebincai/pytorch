import torch
from model import Net
import os
from config import cfg, init_conf
from data import DataSet
from pylib.utils.log import init_logger
import logging


class Tester(object):
    def __init__(self):
        self.device = torch.device("cuda" if cfg.use_cuda else "cpu")
        model_path = os.path.join(cfg.checkpoint, cfg.model_name)
        self.net = Net()
        self.net.load_state_dict(torch.load(model_path))
        self.net.to(self.device)

        self.data_set = DataSet()

    def test(self):
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


if __name__ == "__main__":
    init_conf(cfg)
    init_logger(file_name=cfg.log_name, directory=cfg.checkpoint)
    tester = Tester()
    tester.test()