import os
import utils
import argparse
import torch
import torch.nn as nn
from model import Discriminator, Generator
from data_loader import get_data_loader
from config import get_cfg_defaults


class Trainer(object):
    def __init__(self):
        pass

    def train(self, cfg):
        _com_cfg = cfg.COM
        _train_cfg = cfg.TRAIN

        gen_img_path = os.path.join(_com_cfg.checkpoint, "gen_img")
        if not os.path.exists(gen_img_path):
            os.makedirs(gen_img_path, exist_ok=True)

        dis_net = Discriminator()
        gen_net = Generator(_com_cfg.z_dimension)
        train_loader = get_data_loader(cfg, "train")
        criterion = nn.BCELoss()

        dis_optim = torch.optim.Adam(dis_net.parameters(), lr=_train_cfg.lr)
        gen_optim = torch.optim.Adam(gen_net.parameters(), lr=_train_cfg.lr)

        gen_epoches = 1
        for epoch_idx in range(_train_cfg.epochs):
            for iter_idx, (imgs, labels) in enumerate(train_loader):
                real_imgs = imgs.view(_train_cfg.batch_size, -1)
                real_labels = torch.ones(_train_cfg.batch_size)

                fake_labels = torch.zeros(_train_cfg.batch_size)
                # compute loss of real images
                real_out = dis_net(real_imgs)
                dis_loss_real = criterion(real_out, real_labels)
                real_scores = real_out

                # compute loss of fake images
                z_data = torch.randn(_train_cfg.batch_size, _com_cfg.z_dimension)
                fake_images = gen_net(z_data)
                fake_out = dis_net(fake_images)
                dis_loss_fake = criterion(fake_out, fake_labels)
                fake_scores = fake_out

                dis_loss = dis_loss_real + dis_loss_fake
                dis_optim.zero_grad()
                dis_loss.backward()
                dis_optim.step()

                for j in range(gen_epoches):
                    fake_labels = torch.ones(_train_cfg.batch_size)
                    z_data = torch.randn(_train_cfg.batch_size, _com_cfg.z_dimension)
                    fake_img = gen_net(z_data)
                    output = dis_net(fake_img)
                    g_loss = criterion(output, fake_labels)
                    gen_optim.zero_grad()
                    g_loss.backward()
                    gen_optim.step()

                if iter_idx % cfg.TRAIN.log_interval == 0:
                    print('Epoch: {} [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} D real: {:.6f}, D fake: {:.6f}'.format(
                        epoch_idx, iter_idx, len(train_loader), dis_loss.item(), g_loss.item(),
                        real_scores.data.mean(), fake_scores.data.mean()))

            utils.show_img(fake_img, epoch_idx, gen_img_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", default="conf/default.yaml", type=str, help="the yaml config file")
    args, un_parsed = parser.parse_known_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)

    os.makedirs(cfg.COM.checkpoint, exist_ok=True)
    trainer = Trainer()
    trainer.train(cfg)
