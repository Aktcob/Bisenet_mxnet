import os,sys
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import symbol
from mxnet.gluon import nn
from mxnet import initializer
from mxnet import ndarray as nd
from gluoncv.model_zoo.resnetv1b import resnet18_v1b

#ARM
class ARM(nn.HybridBlock):
    def __init__(self,in_channels,**kwargs):
        super(ARM,self).__init__(**kwargs)
        self.pool = nn.GlobalAvgPool2D()
        self.conv = nn.Conv2D(in_channels=in_channels,channels=in_channels,kernel_size=1,strides=1)
        self.bn = nn.BatchNorm()
    def hybrid_forward(self,F,x):
        in_ = x
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = F.sigmoid(x)
        out = x*in_
        return out

# define Conv-Bn-Relu function
def ConvBnRelu(in_channels,out_channels,kernel_size,strides,padding):
    out = nn.HybridSequential()
    with out.name_scope():
        out.add(nn.Conv2D(in_channels=in_channels,channels=out_channels,kernel_size=kernel_size,strides=strides,padding=padding,use_bias=False))
        out.add(nn.BatchNorm())
        out.add(nn.Activation('relu'))
    return out

#FFM
class FFM(nn.HybridBlock):
    def __init__(self,in_channels,**kwargs):
        super(FFM,self).__init__(**kwargs)
        self.block = ConvBnRelu(in_channels,in_channels,3,1,1)
        self.pool = nn.GlobalAvgPool2D()
        self.conv1 = nn.Conv2D(in_channels=in_channels,channels=in_channels,kernel_size=1,strides=1)
        self.conv2 = nn.Conv2D(in_channels=in_channels,channels=in_channels,kernel_size=1,strides=1)
    def hybrid_forward(self,F,x,s):
        fusion_in = F.concat(x,s)
        fusion_in = self.block(fusion_in)
        x = self.pool(fusion_in)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.sigmoid(x)
        out = fusion_in + fusion_in*x
        return out

#Spatial
class Spatial(nn.HybridBlock):
    def __init__(self,**kwargs):
        super(Spatial,self).__init__(**kwargs)
        self.layer1 = ConvBnRelu(3,64,7,2,3)
        self.layer2 = ConvBnRelu(64,128,3,2,1)
        self.layer3 = ConvBnRelu(128,128,3,2,1)
        self.layer4 = ConvBnRelu(128,256,1,1,0)
    def hybrid_forward(self,F,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def UnSampling(F,x,h,w):
    return F.contrib.BilinearResize2D(x,height = h,width = w)


# Context
class Context(nn.HybridBlock):
    def __init__(self,**kwargs):
        super(Context,self).__init__(**kwargs)
        pretrained = resnet18_v1b(pretrained=True, ctx=mx.gpu(0))
        self.conv1 = pretrained.conv1
        self.bn1 = pretrained.bn1
        self.relu = pretrained.relu
        self.maxpool = pretrained.maxpool
        self.layer1 = pretrained.layer1
        self.layer2 = pretrained.layer2
        self.layer3 = pretrained.layer3
        self.layer4 = pretrained.layer4
        self.feature16_ARM = ARM(256)
        self.feature16_ARM.initialize(ctx=mx.gpu(0))
        self.feature32_ARM = ARM(512)
        self.feature32_ARM.initialize(ctx=mx.gpu(0))
        self.pool = nn.GlobalAvgPool2D()

    def hybrid_forward(self,F,x):
        _,_,h,w = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        f16 = self.layer3(x)
        f32 = self.layer4(f16)
        global_feature = self.pool(f32)
        f16 = self.feature16_ARM(f16)
        f32 = self.feature32_ARM(f32)
        global_feature_up = F.contrib.BilinearResize2D(global_feature,height = h//32,width = w//32)
        ARM32 = global_feature_up + f32
        ARM32_up = F.contrib.BilinearResize2D(ARM32,height = h//8,width = w//8)
        ARM16_up = F.contrib.BilinearResize2D(f16,height = h//8,width = w//8)
        out = F.concat(ARM16_up,ARM32_up)
        return out

class Bisenet(nn.HybridBlock):
    def __init__(self,class_num,channels = 1024,**kwargs):
        super(Bisenet,self).__init__(**kwargs)
        self.spatial = Spatial()
        self.spatial.initialize(init=initializer.Xavier(),ctx=mx.gpu(0))
        self.context = Context()
        self.ffm = FFM(channels)
        self.ffm.initialize(init=initializer.Xavier(),ctx=mx.gpu(0))
        self.conv = nn.Conv2D(in_channels=channels,channels=class_num,kernel_size=1,strides=1)
        self.conv.initialize(init=initializer.Xavier(),ctx=mx.gpu(0))
    def hybrid_forward(self,F,x):
        _,_,h,w = x.shape
        s_path = self.spatial(x)
        c_path = self.context(x)
        ffm_ = self.ffm(s_path,c_path)
        pre_out = F.contrib.BilinearResize2D(ffm_,height = h,width = w)
        out = self.conv(pre_out)
        outputs = []
        outputs.append(out)
        return tuple(outputs)

    # def evaluate(self, x):
    #     """evaluating network with inputs and targets"""
    #     return self.forward(x)[0]




