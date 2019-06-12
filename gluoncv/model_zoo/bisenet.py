from __future__ import division
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.context import cpu
from mxnet.gluon.nn import HybridBlock
from .segbase import SegBaseModel
from .resnetv1b import resnet18_v1b

__all__ = ['bisenet', 'get_bisenet']

# nearest upsampling in context path
def _upsample(x, stride=2):
    """Simple upsampling layer by stack pixel alongside horizontal and vertical directions.
    Parameters
    ----------
    x : mxnet.nd.NDArray or mxnet.symbol.Symbol
        The input array.
    stride : int, default is 2
        Upsampling stride
    """
    return x.repeat(axis=-1, repeats=stride).repeat(axis=-2, repeats=stride)

# convlution // batch normalization // Relu
class CBR(HybridBlock):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pad, 
                                    norm_layer=nn.BatchNorm, norm_kwargs=None, 
                                    is_bn=True, is_relu=True, is_bias=False):
        super(CBR, self).__init__()
        self.is_bn = is_bn
        self.is_relu = is_relu
        with self.name_scope():
            self.conv = nn.Conv2D(in_channels=in_channels, channels=out_channels, 
                    kernel_size=kernel_size, strides=stride, padding=pad, use_bias=is_bias)
            if self.is_bn:
                self.bn = norm_layer(in_channels=out_channels, **({} if norm_kwargs is None else norm_kwargs))
            if self.is_relu:
                self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        if self.is_bn:
            x = self.bn(x)
        if self.is_relu:
            x = self.relu(x)
        return x
        

class spatialpath(HybridBlock):
    def __init__(self, norm_layer, norm_kwargs):
        super(spatialpath, self).__init__()
        with self.name_scope():
            self.conv1 = CBR(3, 64, 7, 2, 3, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv2 = CBR(64, 64, 3, 2, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv3 = CBR(64, 64, 3, 2, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv4 = CBR(64, 128, 1, 1, 0, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        
    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        output = self.conv4(x)
        return output


class ARM(HybridBlock):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs):    
        super(ARM, self).__init__()
        with self.name_scope():
            self.conv1 = CBR(in_channels, out_channels, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv2 = CBR(out_channels, out_channels, 1, 1, 0, norm_layer=norm_layer, norm_kwargs=norm_kwargs, is_relu=False)
            self.act = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        w = self.conv2(w)
        w = self.act(w)
        return F.broadcast_mul(x, w)

class FFM(HybridBlock):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs):
        super(FFM, self).__init__()
        with self.name_scope():
            self.conv1 = CBR(in_channels, out_channels, 1, 1, 0, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv2 = CBR(out_channels, out_channels // 4, 1, 1, 0,
                                    norm_layer=norm_layer, norm_kwargs=norm_kwargs, is_bn=False)
            self.conv3 = CBR(out_channels // 4, out_channels, 1, 1, 0, 
                                            norm_layer=norm_layer, norm_kwargs=norm_kwargs, is_relu=False, is_bn=False)
            self.act = nn.Activation('sigmoid')


    def hybrid_forward(self, F, x1, x2):
        feature = F.concat(x1, x2, dim=1)
        feature = self.conv1(feature)
        se = F.contrib.AdaptiveAvgPooling2D(feature, output_size=1)
        se = self.conv2(se)
        se = self.conv3(se)
        se = self.act(se)
        return F.broadcast_add(feature, F.broadcast_mul(feature, se))


class contextpath(HybridBlock):
    def __init__(self, aux=True, pretrained_base=True, ctx=cpu(), norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(contextpath, self).__init__()
        self.aux = aux

        with self.name_scope():
            # backbone
            pretrained = resnet18_v1b(pretrained=pretrained_base, dilated=False, ctx=ctx, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv1 = pretrained.conv1
            self.bn1 = pretrained.bn1
            self.relu = pretrained.relu
            self.maxpool = pretrained.maxpool
            self.layer1 = pretrained.layer1
            self.layer2 = pretrained.layer2
            self.layer3 = pretrained.layer3
            self.layer4 = pretrained.layer4
            self.global_pool = nn.GlobalAvgPool2D()
            self.global_pool.initialize(ctx=ctx)

            self.conv1 = CBR(512, 128, 1, 1, 0, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv1.initialize(ctx=ctx)
            self.conv1.collect_params().setattr('lr_mult', 10)

            self.arm32 = ARM(512, 128, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.arm32.initialize(ctx=ctx)
            self.arm32.collect_params().setattr('lr_mult', 10)

            self.conv32 = CBR(128, 128, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv32.initialize(ctx=ctx)
            self.conv32.collect_params().setattr('lr_mult', 10)

            self.arm16 = ARM(256, 128, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.arm16.initialize(ctx=ctx)
            self.arm16.collect_params().setattr('lr_mult', 10)

            self.conv16 = CBR(128, 128, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv16.initialize(ctx=ctx)
            self.conv16.collect_params().setattr('lr_mult', 10)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        feature8 = self.layer2(x)
        feature16 = self.layer3(feature8)
        feature32 = self.layer4(feature16)

        avg = self.global_pool(feature32)
        avg = self.conv1(avg)

        feature_arm32 = self.arm32(feature32)
        feature32 = F.broadcast_add(avg, feature_arm32)

        feature32 = _upsample(feature32)   # nearest upsampling
        feature32 = self.conv32(feature32)
        
        feature_arm16 = self.arm16(feature16)
        feature16 = F.broadcast_add(feature32, feature_arm16)
        
        feature16 = _upsample(feature16)   # nearest upsampling
        feature16 = self.conv16(feature16)
        
        if self.aux:
            return feature32, feature16
        else:
            return feature16

class head(HybridBlock):
    def __init__(self, nclass, in_channels, out_channels, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(head, self).__init__()
        with self.name_scope():
            self.conv_3x3 = CBR(in_channels, out_channels, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv_1x1 = nn.Conv2D(in_channels=out_channels, channels=nclass, kernel_size=1, strides=1, padding=0)

    def hybrid_forward(self, F, x):
        x = self.conv_3x3(x)
        return self.conv_1x1(x)

class bisenet(HybridBlock):
    def __init__(self, nclass=19, aux=True, ctx=cpu(), pretrained_base=True, crop_size=480, **kwargs):
        super(bisenet, self).__init__()
        self.crop_size = crop_size
        self.aux = aux
        with self.name_scope():
            self.spatial_path = spatialpath(**kwargs)
            self.spatial_path.initialize(ctx=ctx)

            self.context_path = contextpath(aux=self.aux, ctx=ctx, **kwargs)

            self.ffm = FFM(256, 256, **kwargs)
            self.ffm.initialize(ctx=ctx)
            self.ffm.collect_params().setattr('lr_mult', 10)

            self.main_head = head(nclass, 256, 64, **kwargs)
            self.main_head.initialize(ctx=ctx)
            self.main_head.collect_params().setattr('lr_mult', 10)

            if self.aux:
                self.aux1_head = head(nclass, 128, 256, **kwargs)
                self.aux1_head.initialize(ctx=ctx)
                self.aux1_head.collect_params().setattr('lr_mult', 10)

                self.aux2_head = head(nclass, 128, 256, **kwargs)
                self.aux2_head.initialize(ctx=ctx)
                self.aux2_head.collect_params().setattr('lr_mult', 10)


    def hybrid_forward(self, F, x):
        outputs = []

        s_out = self.spatial_path(x)
        if self.aux:
            feature32, feature16 = self.context_path(x)
        else:
            feature16 = self.context_path(x)

        mix = self.ffm(s_out,feature16)
        mix = self.main_head(mix)
        main_out = F.contrib.BilinearResize2D(mix, height=self.crop_size, width=self.crop_size)
        outputs.append(main_out)

        if autograd.is_training():
            if self.aux:
                aux1_head_out = self.aux1_head(feature16)
                aux1_head_out = F.contrib.BilinearResize2D(aux1_head_out, height=self.crop_size, width=self.crop_size)
                outputs.append(aux1_head_out)

                aux2_head_out = self.aux2_head(feature32)
                aux2_head_out = F.contrib.BilinearResize2D(aux2_head_out, height=self.crop_size, width=self.crop_size)
                outputs.append(aux2_head_out)
            return tuple(outputs)

        outputs = outputs[0]
        outputs = F.argmax(outputs,1)
        outputs = F.squeeze(outputs)
        return outputs

    def evaluate(self, x):
        return self.forward(x)

def get_bisenet(nclass, aux=1, crop_size=418, **kwargs):
    net = bisenet(nclass, aux=aux, crop_size=crop_size, **kwargs) 
    return net


