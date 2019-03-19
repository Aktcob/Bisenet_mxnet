import os,sys
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import symbol
from mxnet import autograd
from mxnet.gluon import nn
from mxnet import initializer
from mxnet import ndarray as nd
from gluoncv.model_zoo.resnetv1b import resnet18_v1b,resnet50_v1s

# define Conv-Bn-Relu function
def ConvBnRelu(in_channels,out_channels,kernel_size,strides,padding):
    out = nn.HybridSequential()
    with out.name_scope():
        out.add(nn.Conv2D(in_channels=in_channels,channels=out_channels,kernel_size=kernel_size,strides=strides,padding=padding,use_bias=False))
        out.add(nn.BatchNorm())
        out.add(nn.Activation('relu'))
    return out

#ARM
class ARM(nn.HybridBlock):
    def __init__(self,in_channels,out_channel,**kwargs):
        super(ARM,self).__init__(**kwargs)
        self.conv3 = ConvBnRelu(in_channels,out_channel,3,1,1)
        self.pool = nn.GlobalAvgPool2D()
        self.conv = nn.Conv2D(in_channels=out_channel,channels=out_channel,kernel_size=1,strides=1,padding=0)
        self.bn = nn.BatchNorm()
    def hybrid_forward(self,F,x):
        in_ = self.conv3(x)
        se = self.pool(in_)
        se = self.conv(se)
        se = self.bn(se)
        se = F.sigmoid(se)
        out = F.broadcast_mul(in_, se)
        # out = in_*se
        return out

#FFM
class FFM(nn.HybridBlock):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(FFM,self).__init__(**kwargs)
        self.block = ConvBnRelu(in_channels,out_channels,3,1,1)
        self.pool = nn.GlobalAvgPool2D()
        self.conv1 = nn.Conv2D(in_channels=out_channels,channels=out_channels,kernel_size=1,strides=1)
        self.conv2 = nn.Conv2D(in_channels=out_channels,channels=out_channels,kernel_size=1,strides=1)
    def hybrid_forward(self,F,x,s):
        fusion_in = F.concat(x,s)
        fusion_in = self.block(fusion_in)
        x = self.pool(fusion_in)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.sigmoid(x)
        out = fusion_in + F.broadcast_mul(fusion_in,x)
        return out

#Spatial
class Spatial(nn.HybridBlock):
    def __init__(self,**kwargs):
        super(Spatial,self).__init__(**kwargs)
        self.layer1 = ConvBnRelu(3,64,7,2,3)
        self.layer2 = ConvBnRelu(64,64,3,2,1)
        self.layer3 = ConvBnRelu(64,64,3,2,1)
        self.layer4 = ConvBnRelu(64,128,1,1,0)
    def hybrid_forward(self,F,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

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
        self.feature16_ARM = ARM(256,128)
        self.feature16_ARM.initialize(init=initializer.Xavier(),ctx=mx.gpu(0))
        self.feature32_ARM = ARM(512,128)
        self.feature32_ARM.initialize(init=initializer.Xavier(),ctx=mx.gpu(0))
        self.pool = nn.GlobalAvgPool2D()
        self.conv_p = ConvBnRelu(512,128,1,1,0)
        self.conv_p.initialize(init=initializer.Xavier(),ctx=mx.gpu(0))
        self.aux_conv1 = ConvBnRelu(128,128,3,1,1)
        self.aux_conv1.initialize(init=initializer.Xavier(),ctx=mx.gpu(0))
        self.aux_conv2 = ConvBnRelu(128,128,3,1,1)
        self.aux_conv2.initialize(init=initializer.Xavier(),ctx=mx.gpu(0))

    def hybrid_forward(self,F,x):
        # _,_,h,w = x.shape
        h, w = 1024, 1024
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        f16 = self.layer3(x)
        f32 = self.layer4(f16)
        global_feature = self.pool(f32)
        global_feature = self.conv_p(global_feature)
        f16 = self.feature16_ARM(f16)
        f32 = self.feature32_ARM(f32)
        global_feature_up = F.contrib.BilinearResize2D(global_feature,height = h//32,width = w//32)
        ARM32 = global_feature_up + f32
        ARM32_up = F.contrib.BilinearResize2D(ARM32,height = h//16,width = w//16)
        # if autograd.is_training():
        ARM32_up = self.aux_conv1(ARM32_up)  # aux_loss1

        ARM16 = f16 + ARM32_up
        ARM16_up = F.contrib.BilinearResize2D(ARM16,height = h//8,width = w//8)
        ARM16_up = self.aux_conv2(ARM16_up)
        # if autograd.is_training():
        return ARM32_up, ARM16_up
        # else:
            # return ARM16_up

class BisenetHead(nn.HybridBlock):
    def __init__(self,in_channels,out_channels,choice=False,h=480,w=480,**kwargs):
        super(BisenetHead,self).__init__(**kwargs)
        self.h = h
        self.w = w
        if choice:
            self.conv1 = ConvBnRelu(in_channels,128,3,1,1)
            self.conv2 = ConvBnRelu(128,out_channels,1,1,0)
        else:
            self.conv1 = ConvBnRelu(in_channels,64,3,1,1)
            self.conv2 = ConvBnRelu(64,out_channels,1,1,0)
    
    def hybrid_forward(self,F,x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = F.contrib.BilinearResize2D(x,height = self.h,width = self.w)
        return out



class Bisenet(nn.HybridBlock):
    def __init__(self,class_num,channels = 256,aux=True,**kwargs):
        super(Bisenet,self).__init__(**kwargs)
        self.aux = aux
        w,h =1024,1024
        self.spatial = Spatial()
        self.spatial.initialize(init=initializer.Xavier(),ctx=mx.gpu(0))
        self.context = Context()
        self.ffm = FFM(256,channels)
        self.ffm.initialize(init=initializer.Xavier(),ctx=mx.gpu(0))
        self.mainhead = BisenetHead(256,class_num,False,w,h)
        self.mainhead.initialize(init=initializer.Xavier(),ctx=mx.gpu(0))
        if self.aux:
            self.aux1head = BisenetHead(128,class_num,True,w,h)
            self.aux1head.initialize(init=initializer.Xavier(),ctx=mx.gpu(0))
            self.aux2head = BisenetHead(128,class_num,True,w,h)
            self.aux2head.initialize(init=initializer.Xavier(),ctx=mx.gpu(0))
    def hybrid_forward(self,F,x):
        outputs = []
        # _,_,h,w = x.shape
        s_path = self.spatial(x)
        # print "s_path",s_path.shape
        c32_path, c16_path = self.context(x)
        ffm_ = self.ffm(s_path,c16_path)
        out = self.mainhead(ffm_)
        if self.aux:
            aux1 = self.aux1head(c16_path)
            aux2 = self.aux2head(c32_path)
            outputs.append(aux1)
            outputs.append(aux2)
            outputs.append(out)
        if autograd.is_training(): 
            return tuple(outputs)
        else:
            outputs.append(F.squeeze(F.argmax(out, 1)))
            return tuple(outputs)
            # return tuple(out)



if __name__ == '__main__':
    image = nd.random.normal(shape=(2,3,480,480),ctx=mx.gpu(0))
    net = Bisenet(19)
    # x = net(image)
    # print net
    print(net(image))


