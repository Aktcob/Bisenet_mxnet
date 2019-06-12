"""
This module provides data loaders and transfomers for popular vision datasets.
"""
from . import transforms
from . import batchify
from .imagenet.classification import ImageNet, ImageNet1kAttr
from .dataloader import DetectionDataLoader, RandomTransformDataLoader

from .cityscapes import CitySegmentation

from .segbase import ms_batchify_fn


