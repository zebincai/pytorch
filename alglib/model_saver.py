import torch
import os


def saver(model, path, name):
    mode_file = os.path.join(path, name)
    torch.save(model.state_dict(), mode_file)
