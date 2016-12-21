#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers


#optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)

#入力ベクトルが1000, 出力は2値
class ClassificationModel(chainer.FunctionSet):
    def __init__(self, f_out, s_out):
        super(ClassificationModel, self).__init__(
                l1=L.Linear(1000, f_out), #Linear: 全結合
                l2=L.Linear(f_out, s_out), #4層
                l3=L.Linear(s_out, 2)
        )

    def __call__(self, x, t, train):
        #データの保持
        x = chainer.Variable(x)
        t = chainer.Variable(t)

        #活性化関数: 出力の正規化
        h = F.relu(self.l1(x)) #活性化関数 relu: 正規化線形関数 f(x) = max(0, x)
        h = F.relu(self.l2(h))
        h = self.l3(h)

        if train:
            #誤差関数:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
        else:
            return F.accuracy(h, t)


def set_optimizer(opt):
    if opt == "SGD":
        return optimizers.SGD()
    if opt == "MomentumSGD":
        return optimizers.MomentumSGD()
    if opt == "AdaGrad":
        return optimizers.AdaGrad()
    if opt == "AdaDelta":
        return optimizers.AdaDelta()
    if opt == "Adam":
        return optimizers.Adam()
    if opt == "NesterovAG":
        return optimizers.NesterovAG()
    if opt == "RMSprop":
        return optimizers.RMSprop()
    if opt == "RMSpropGraves":
        return optimizers.RMSpropGraves()
    if opt == "SMORMS3":
        return optimizers.SMORMS3()
    print "error"
    return None

def nn(train_label, train_data, test_label, test_data):
    train_data = train_data.astype(np.float32)
    train_label = train_label.astype(np.int32)
    test_data = test_data.astype(np.float32)
    test_label = test_label.astype(np.int32)
    for f_out in [n * 100 for n in range(1, 10)]:
        for s_out in [n * 100 for n in range(1, 10)]:
            #for opt in ["SGD", "MomentumSGD", "AdaGrad", "AdaDelta", "Adam", "NesterovAG", "RMSpropGraves", "SMORMS3", "NesterovAG"]:
            for opt in ["AdaGrad"]:
                model = ClassificationModel(f_out, s_out)
                #optimizer = optimizers.Adam()
                optimizer = set_optimizer(opt)
                optimizer.setup(model)

                for epoch in range(500):
                    #勾配を0初期化
                    model.zerograds()
                    #学習,及び誤差関数の結果と精度の出力
                    loss, acc = model(train_data, train_label, train=True)
                    #逆伝搬による最適化
                    loss.backward()
                    optimizer.update()
                    #print epoch, "acc  ", acc.data

                acc = model(test_data, test_label, train=False)
                print opt, f_out, s_out, "acc test ", acc.data
                model = optimizer = None
                # global optimizer
                #
                # model = ClassificationModel()
                # optimizer.setup(model.collect_parameters())
                # for indata, label in zip(data, labels):
                #     model.train(indata, label)
                #     print label
