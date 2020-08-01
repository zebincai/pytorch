import os
import torch
import torch.optim as optim
from model import Net
from config import get_cfg_defaults
from loss import CELoss
from data import get_data_loader
from pylib.utils.log import init_logger
import argparse
import logging


class Trainer(object):
    def __init__(self):
        self.valid_loader = None

    def eval(self, cfg, net):
        if self.valid_loader is None:
            self.valid_loader = get_data_loader(cfg)
        net.eval()
        correct = 0
        with torch.no_grad():
            for data, target in self.valid_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = net(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        logging.info('\nTest set:Accuracy: {}/{} ({:.02f}%)\n'.format(
            correct, len(self.data_set.test_loader.dataset),
            100. * correct / len(self.data_set.test_loader.dataset)))

    def train(self, cfg):
        device = torch.device("cuda" if cfg.use_cuda else "cpu")
        net = Net().to(device)
        train_loader = get_data_loader(cfg, phase="train")
        loss_func = CELoss()
        optimizer = optim.SGD(params=net.parameters(),
                              lr=cfg.TRAIN.lr,
                              momentum=cfg.TRAIN.momentum)
        for epoch in range(1, cfg.TRAIN.epochs + 1):
            # set training model
            net.train()

            for batch_idx, (data, target) in enumerate(train_loader):
                # load batch data and set the corresponding device mode
                data = data.to(device)
                target = target.to(device)

                # zero the gradient
                optimizer.zero_grad()
                # infer the data
                output = net(data)
                # loss calculation
                loss = loss_func(output, target)
                # gradient  calculation
                loss.backward()
                # back propagation
                optimizer.step()

                if batch_idx % cfg.TRAIN.log_interval == 0:
                    logging.info('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader), loss.item()))

            model_file = os.path.join(cfg.checkpoint, cfg.model_name)
            net.save_model(model_file)


if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", default="cfg/v1.yaml", type=str, help="the yaml config file")
    args, un_parsed = parser.parse_known_args()

    # parse config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)
    opts = ["TRAIN.batch_size", 32,
            "TRAIN.epochs", 5]

    cfg.merge_from_list(opts)
    cfg.freeze()

    # init log file
    os.makedirs(cfg.checkpoint, exist_ok=True)
    log_file_name = os.path.basename(args.cfg_file).replace(".yaml", ".log")
    init_logger(file_name=log_file_name, directory=cfg.checkpoint)
    logging.info(cfg)
    trainer = Trainer()
    trainer.train(cfg)
