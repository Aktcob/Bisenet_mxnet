# pylint: disable=wildcard-import, unused-wildcard-import
"""Model store which handles pretrained models from both
mxnet.gluon.model_zoo.vision and gluoncv.models
"""
from .fcn import *
from .pspnet import *
from .deeplabv3 import *
from .resnetv1b import *
from .resnext import *
from .resnet import *
from .bisenet import *

__all__ = ['get_model', 'get_model_list']

_models = {
    'resnet18_v1': resnet18_v1,
    'resnet34_v1': resnet34_v1,
    'resnet50_v1': resnet50_v1,
    'resnet101_v1': resnet101_v1,
    'resnet152_v1': resnet152_v1,
    'resnet18_v2': resnet18_v2,
    'resnet34_v2': resnet34_v2,
    'resnet50_v2': resnet50_v2,
    'resnet101_v2': resnet101_v2,
    'resnet152_v2': resnet152_v2,
    'fcn_resnet50_voc': get_fcn_resnet50_voc,
    'fcn_resnet101_coco': get_fcn_resnet101_coco,
    'fcn_resnet101_voc': get_fcn_resnet101_voc,
    'fcn_resnet50_ade': get_fcn_resnet50_ade,
    'fcn_resnet101_ade': get_fcn_resnet101_ade,
    'psp_resnet101_coco': get_psp_resnet101_coco,
    'psp_resnet101_voc': get_psp_resnet101_voc,
    'psp_resnet50_ade': get_psp_resnet50_ade,
    'psp_resnet101_ade': get_psp_resnet101_ade,
    'psp_resnet101_citys': get_psp_resnet101_citys,
    'deeplab_resnet101_coco': get_deeplab_resnet101_coco,
    'deeplab_resnet101_voc': get_deeplab_resnet101_voc,
    'deeplab_resnet152_coco': get_deeplab_resnet152_coco,
    'deeplab_resnet152_voc': get_deeplab_resnet152_voc,
    'deeplab_resnet50_ade': get_deeplab_resnet50_ade,
    'deeplab_resnet101_ade': get_deeplab_resnet101_ade,
    'resnet18_v1b': resnet18_v1b,
    'resnet34_v1b': resnet34_v1b,
    'resnet50_v1b': resnet50_v1b,
    'resnet101_v1b': resnet101_v1b,
    'resnet152_v1b': resnet152_v1b,
    'resnet50_v1c': resnet50_v1c,
    'resnet101_v1c': resnet101_v1c,
    'resnet152_v1c': resnet152_v1c,
    'resnet50_v1d': resnet50_v1d,
    'resnet101_v1d': resnet101_v1d,
    'resnet152_v1d': resnet152_v1d,
    'resnet50_v1e': resnet50_v1e,
    'resnet101_v1e': resnet101_v1e,
    'resnet152_v1e': resnet152_v1e,
    'resnet50_v1s': resnet50_v1s,
    'resnet101_v1s': resnet101_v1s,
    'resnet152_v1s': resnet152_v1s,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x4d': resnext101_32x4d,
    'resnext101_64x4d': resnext101_64x4d,
    'bisenet': get_bisenet,
    }

def get_model(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    classes : int
        Number of classes for the output layer.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    HybridBlock
        The model.
    """
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net

def get_model_list():
    """Get the entire list of model names in model_zoo.

    Returns
    -------
    list of str
        Entire list of model names in model_zoo.

    """
    return _models.keys()
