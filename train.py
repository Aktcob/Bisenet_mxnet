import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm

import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.loss import *
from gluoncv.utils import LRScheduler
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo.bisenet import *
from gluoncv.model_zoo import get_model
from gluoncv.utils.parallel import *
from gluoncv.data import CitySegmentation
from gluoncv import utils as gutils

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='MXNet Gluon \
                                     Segmentation')
    # model and dataset 
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    parser.add_argument('--train-split', type=str, default='train',
                        help='dataset train split (default: train)')
    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxilary loss')
    parser.add_argument('--aux-weight', type=float, default=0.5,
                        help='auxilary loss weight')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, \
                        and beta/gamma for batchnorm layers.')
    # cuda and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--ngpus', type=int,
                        default=len(mx.test_utils.list_gpus()),
                        help='number of GPUs (default: 4)')

    parser.add_argument('--kvstore', type=str, default='device',
                        help='kvstore to use for trainer/module.')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='default',
                        help='set the checkpoint name')
    # evaluation only
    parser.add_argument('--eval', action='store_true', default= False,
                        help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default= False,
                            help='skip validation during training')
    # synchronized Batch Normalization
    parser.add_argument('--syncbn', action='store_true', default= False,
                        help='using Synchronized Cross-GPU BatchNorm')
    # the parser
    args = parser.parse_args()
    # handle contexts
    if args.no_cuda:
        print('Using CPU')
        args.kvstore = 'local'
        args.ctx = [mx.cpu(0)]
    else:
        print('Number of GPUs:', args.ngpus)
        args.ctx = [mx.gpu(i) for i in range(args.ngpus)]
    # Synchronized BatchNorm
    args.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if args.syncbn \
        else mx.gluon.nn.BatchNorm
    args.norm_kwargs = {'num_devices': args.ngpus} if args.syncbn else {}
    print(args)
    return args

class Trainer(object):
    def __init__(self, args):
        self.args = args
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader 
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                       'crop_size': args.crop_size}

        trainset = CitySegmentation(split=args.train_split, mode='train', **data_kwargs)
        valset = CitySegmentation(split='val', mode='val', **data_kwargs)
        num_class = CitySegmentation.NUM_CLASS

        self.train_data = gluon.data.DataLoader(
            trainset, args.batch_size, shuffle=True, last_batch='rollover',
            num_workers=args.workers)
        self.eval_data = gluon.data.DataLoader(valset, args.test_batch_size,
            last_batch='rollover', num_workers=args.workers)

        model = get_bisenet(num_class, aux=args.aux, crop_size=args.crop_size, 
                                        norm_layer=args.norm_layer, norm_kwargs=args.norm_kwargs)                        
        model.cast(args.dtype)
        # create network 

        self.net = DataParallelModel(model, args.ctx, args.syncbn)
        self.evaluator = DataParallelModel(SegEvalModel(model), args.ctx)
        # resume checkpoint if needednvidia-smi
        if args.resume is not None:
            if os.path.isfile(args.resume):
                print('load parameters!')
                model.load_parameters(args.resume, ctx=args.ctx)
            else:
                raise RuntimeError("=> no checkpoint found at '{}'" \
                    .format(args.resume))
        # create criterion
        criterion = MixSoftmaxCrossEntropyLoss(args.aux, aux_weight=args.aux_weight)  # OHEM loss func does not exist right now.
        self.criterion = DataParallelCriterion(criterion, args.ctx, args.syncbn)
        # optimizer and lr scheduling
        self.lr_scheduler = LRScheduler(mode='poly', baselr=args.lr,
                                        niters=len(self.train_data), 
                                        nepochs=args.epochs,step=(30, 40, 50))
        kv = mx.kv.create(args.kvstore)
        optimizer_params = {'lr_scheduler': self.lr_scheduler,
                            'wd':args.weight_decay,
                            'momentum': args.momentum}
        if args.dtype == 'float16':
            optimizer_params['multi_precision'] = True

        if args.no_wd:
            for k, v in self.net.module.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        self.optimizer = gluon.Trainer(self.net.module.collect_params(), 'sgd',
                                       optimizer_params, kvstore = kv)
        # evaluation metrics
        self.metric = gluoncv.utils.metrics.SegmentationMetric(trainset.num_class)

    def training(self, epoch):
        tbar = tqdm(self.train_data)
        train_loss = 0.0
        alpha = 0.2
        for i, (data, target) in enumerate(tbar):
            self.lr_scheduler.update(i, epoch, args.epochs)
            with autograd.record(True):
                outputs = self.net(data.astype(args.dtype, copy=False))
                losses = self.criterion(outputs, target)
                mx.nd.waitall()
                autograd.backward(losses)
            self.optimizer.step(self.args.batch_size)
            for loss in losses:
                train_loss += loss.asnumpy()[0] / len(losses)
            tbar.set_description('Epoch %d, training loss %.3f'%\
                (epoch, train_loss/(i+1)))
            mx.nd.waitall()

        # save every epoch
        save_checkpoint(self.net.module, self.args, False)

    def validation(self, epoch):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        self.metric.reset()
        tbar = tqdm(self.eval_data)
        for i, (data, target) in enumerate(tbar):
            outputs = self.evaluator(data.astype(args.dtype, copy=False))
            outputs = [x[0] for x in outputs]
            targets = mx.gluon.utils.split_and_load(target, args.ctx, even_split=False)
            self.metric.update(targets, outputs)
            pixAcc, mIoU = self.metric.get()
            tbar.set_description('Epoch %d, validation pixAcc: %.3f, mIoU: %.3f'%\
                (epoch, pixAcc, mIoU))
            mx.nd.waitall()


def save_checkpoint(net, args, is_best=False):
    """Save Checkpoint"""
    filename= 'checkpoint.params'
    net.save_parameters(filename)


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    if args.eval:
        print('Evaluating model: ', args.resume)
        trainer.validation(args.start_epoch)
    else:
        print('Starting Epoch:', args.start_epoch)
        print('Total Epoches:', args.epochs)
        for epoch in range(args.start_epoch, args.epochs):
            trainer.training(epoch)
            if not trainer.args.no_val:
                trainer.validation(epoch)
