from yacs.config import CfgNode as CN
import os

_C = CN()
_C.COM = CN()
_C.COM.z_dimension = 100
_C.COM.checkpoint = "D:/workspace/output/gandc"


_C.TRAIN = CN()
_C.TRAIN.batch_size = 100
_C.TRAIN.n_workers = 2
_C.TRAIN.lr = 0.0003
_C.TRAIN.momentum = 0.5
_C.TRAIN.epochs = 10
_C.TRAIN.log_interval = 40    # every interval iteration
_C.TRAIN.use_cuda = True

_C.DATA = CN()
_C.DATA.train_path = "D:/workspace/dataset"
_C.DATA.valid_path = "D:/workspace/dataset"
_C.DATA.test_path = ""


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()