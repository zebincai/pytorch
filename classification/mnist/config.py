from yacs.config import CfgNode as CN

_C = CN()

_C.use_cuda = True
_C.checkpoint = "./checkpoint"
_C.model_name = "mnist_cnn.pt"

_C.DEBUG = CN()
_C.DEBUG.print = True

_C.TRAIN = CN()
_C.TRAIN.batch_size = 64
_C.TRAIN.lr = 0.01
_C.TRAIN.momentum = 0.5
_C.TRAIN.epochs = 10
_C.TRAIN.log_interval = 40    # every interval iteration


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


# Alternatively, provide a way to import the defaults as
# a global singleton:
cfg = _C     # users can `from config import cfg`

