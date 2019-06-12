"""GluonCV Model Zoo"""
# pylint: disable=wildcard-import
from .model_zoo import get_model, get_model_list
from .model_store import pretrained_model_list
from .fcn import *
from .pspnet import *
from .deeplabv3 import *
from . import segbase
from .resnetv1b import *
from .resnet import *
from .bisenet import *
