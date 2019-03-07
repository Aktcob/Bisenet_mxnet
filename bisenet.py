import os,sys
import numpy as np
from mxnet import gluon
from mxnet import symbol
from mxnet.gluon import nn
from mxnet import initializer
from mxnet import ndarray as nd
from gluoncv.model_zoo.resnetv1b import resnet18_v1b




def ConvBnRelu(in_channels,out_channels,kernel_size,strides,padding):
    out = nn.HybridSequential()
    with out.name_scope():
        out.add(nn.Conv2D(in_channels=in_channels,channels=out_channels,kernel_size=kernel_size,strides=strides,padding=padding,use_bias=False))
        out.add(nn.BatchNorm())
        out.add(nn.Activation('relu'))
    return out

def UnSampling(F,x,h,w):
    return F.contrib.BilinearResize2D(x,height = h,width = w)

#Spatial Path
class SpatialPath(nn.HybridBlock):
    def __init__(self,**kwargs):
        super(SpatialPath,self).__init__(**kwargs)
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

#ARM
class AttentionReffineMoudle(nn.HybridBlock):
    def __init__(self,channels,**kwargs):
        super(AttentionReffineMoudle,self).__init__(**kwargs)
        self.pool = nn.GlobalAvgPool2D()
        self.conv = nn.Conv2D(in_channels=channels,channels=channels,kernel_size=1,strides=1)
        self.norm = nn.BatchNorm()
    def hybrid_forward(self,F,x):
        feature_inputs = x
        x = self.pool(x)
        x = self.conv(x)
        x = self.norm(x)
        x = F.sigmoid(x)
        out = x*feature_inputs
        return out

#FFM
class FeatureFusionMoudle(nn.HybridBlock):
    def __init__(self,channels,**kwargs):
        super(FeatureFusionMoudle,self).__init__(**kwargs)
        self.block = nn.HybridSequential()
        with self.block.name_scope():
            self.block.add(nn.Conv2D(in_channels=channels,channels=channels,kernel_size=3,strides=1,padding=1))
            self.block.add(nn.BatchNorm())
            self.block.add(nn.Activation('relu'))
        self.pool = nn.GlobalAvgPool2D()
        self.conv1 = nn.Conv2D(in_channels=channels,channels=channels,kernel_size=1,strides=1)
        self.conv2 = nn.Conv2D(in_channels=channels,channels=channels,kernel_size=1,strides=1)
    def hybrid_forward(self,F,x,s):
        feature = F.concat(x,s)
        feature = self.block(feature)
        x = self.pool(feature)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.sigmoid(x)
        mul = feature*x
        out = feature + mul
        return out

class ContextPath(nn.HybridBlock):
    def __init__(self,**kwargs):
        super(ContextPath,self).__init__(**kwargs)
        pretrained = resnet18_v1b(pretrained=True, ctx=mx.gpu(0))
        self.conv1 = pretrained.conv1
        self.bn1 = pretrained.bn1
        self.relu = pretrained.relu
        self.maxpool = pretrained.maxpool
        self.layer1 = pretrained.layer1
        self.layer2 = pretrained.layer2
        self.layer3 = pretrained.layer3
        self.layer4 = pretrained.layer4
        self.feature_16x_down = AttentionReffineMoudle(256)
        self.feature_16x_down.initialize(ctx=mx.gpu(0))
        self.feature_32x_down = AttentionReffineMoudle(512)
        self.feature_32x_down.initialize(ctx=mx.gpu(0))
        self.pool = nn.GlobalAvgPool2D()
    def hybrid_forward(self,F,x):
        _,_,h,w = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feature_x16 = self.layer3(x)
        feature_x32 = self.layer4(feature_x16)
        global_feature = self.pool(feature_x32)
        feature_x16 = self.feature_16x_down(feature_x16)
        feature_x32 = self.feature_32x_down(feature_x32)
        global_feature_up = F.contrib.BilinearResize2D(global_feature,height = h//32,width = w//32)
        net_5_ARM = global_feature_up + feature_x32

        _,_,h1,w1 = net_5_ARM.shape
        net_5_up = UnSampling(F,net_5_ARM,h1*4,w1*4)
        #net_5_up = UnSampling(F,net_5_ARM,28,28)
        _,_,h2,w2 = feature_x16.shape
        feature_x16_up = UnSampling(F,feature_x16,h2*2,w2*2)
        context_net = F.concat(feature_x16_up,net_5_up)
        return context_net

class BiseNet(nn.HybridBlock):
    def __init__(self,class_num,channels = 1024,**kwargs):
        super(BiseNet,self).__init__(**kwargs)
        self.spatial_path = SpatialPath()
        self.spatial_path.initialize(init=initializer.Xavier(),ctx=mx.gpu(0))
        self.context_path = ContextPath()
        self.ffm = FeatureFusionMoudle(channels)
        self.ffm.initialize(init=initializer.Xavier(),ctx=mx.gpu(0))
        self.pred = nn.Conv2D(in_channels=channels,channels=class_num,kernel_size=1,strides=1)
        self.pred.initialize(init=initializer.Xavier(),ctx=mx.gpu(0))
    def hybrid_forward(self,F,x):
        x1 = self.spatial_path(x)
        x2 = self.context_path(x)
        feature = self.ffm(x1,x2)
        _,_,h,w = feature.shape
        feature_up = UnSampling(F,feature,h*8,w*8)
        #feature_up = UnSampling(F,feature,224,224)
        pred_feature = self.pred(feature_up)
        outputs = []
        outputs.append(pred_feature)
        return tuple(outputs)
    # def evaluate(self, x):
    #     """evaluating network with inputs and targets"""
    #     return self.forward(x)[0]

if __name__ == '__main__':
    image = nd.random.normal(shape=(2,3,480,480))
    net = BiseNet(2)
    print(net(image))




