"""Base Model for Semantic Segmentation"""
import math
import numpy as np
import mxnet as mx
from mxnet.ndarray import NDArray
from mxnet.gluon.nn import HybridBlock
from ..utils.parallel import parallel_apply
from .resnetv1b import resnet50_v1s, resnet101_v1s, resnet18_v1b
from ..utils.parallel import tuple_map
# pylint: disable=wildcard-import,abstract-method,arguments-differ,dangerous-default-value,missing-docstring 

__all__ = ['get_segmentation_model', 'SegBaseModel', 'SegEvalModel']

def get_segmentation_model(model, **kwargs):
    from .bisenet import get_bisenet
    models = {
        'bisenet': get_bisenet,
    }
    return models[model](**kwargs)

class SegBaseModel(HybridBlock):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resn et152').
    norm_layer : Block
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    """
    # pylint : disable=arguments-differ
    def __init__(self, nclass, aux, backbone='resnet50', height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=True, **kwargs):
        super(SegBaseModel, self).__init__()
        self.aux = aux
        self.nclass = nclass
        with self.name_scope():
            if backbone == 'resnet50':
                pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=True, **kwargs)
            elif backbone == 'resnet101':
                pretrained = resnet101_v1s(pretrained=pretrained_base, dilated=True, **kwargs)
            elif backbone == 'resnet18':
                pretrained = resnet18_v1b(pretrained=pretrained_base, dilated=True, **kwargs)
            else:
                raise RuntimeError('unknown backbone: {}'.format(backbone))
            self.conv1 = pretrained.conv1
            self.bn1 = pretrained.bn1
            self.relu = pretrained.relu
            self.maxpool = pretrained.maxpool
            self.layer1 = pretrained.layer1
            self.layer2 = pretrained.layer2
            self.layer3 = pretrained.layer3
            self.layer4 = pretrained.layer4
        height = height if height is not None else crop_size
        width = width if width is not None else crop_size
        self._up_kwargs = {'height': height, 'width': width}
        self.base_size = base_size
        self.crop_size = crop_size

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        c3 = self.layer3(x)
        c4 = self.layer4(c3)
        return c3, c4

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        h, w = x.shape[2:]
        self._up_kwargs['height'] = h
        self._up_kwargs['width'] = w
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred


class SegEvalModel(object):
    """Segmentation Eval Module"""
    def __init__(self, module):
        self.module = module

    def __call__(self, *inputs, **kwargs):
        return self.module.evaluate(*inputs, **kwargs)

    def collect_params(self):
        return self.module.collect_params()


def _resize_image(img, h, w):
    return mx.nd.contrib.BilinearResize2D(img, height=h, width=w)


def _pad_image(img, crop_size=480):
    b, c, h, w = img.shape
    assert(c == 3)
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    mean = [.485, .456, .406]
    std = [.229, .224, .225]
    pad_values = -np.array(mean) / np.array(std)
    img_pad = mx.nd.zeros((b, c, h + padh, w + padw)).as_in_context(img.context)
    for i in range(c):
        img_pad[:, i, :, :] = mx.nd.squeeze(
            mx.nd.pad(img[:, i, :, :].expand_dims(1), 'constant',
                      pad_width=(0, 0, 0, 0, 0, padh, 0, padw),
                      constant_value=pad_values[i]
                     ))
    assert(img_pad.shape[2] >= crop_size and img_pad.shape[3] >= crop_size)
    return img_pad


def _crop_image(img, h0, h1, w0, w1):
    return img[:, :, h0:h1, w0:w1]


def _flip_image(img):
    assert(img.ndim == 4)
    return img.flip(3)
