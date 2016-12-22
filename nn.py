#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers

import threading

#入力ベクトルが1000, 出力は2値
class ClassificationModel(chainer.FunctionSet):
    #中間層は2つ。計４層
    def __init__(self, h1_unit, h2_unit):
        super(ClassificationModel, self).__init__(
                l1=L.Linear(1000, h1_unit), #Linear: 全結合
                l2=L.Linear(h1_unit, h2_unit),
                l3=L.Linear(h2_unit, 2)
        )

    def __call__(self, x, t, train):
        #データの保持
        x = chainer.Variable(x)
        t = chainer.Variable(t)

        #活性化関数: 出力の正規化
        #relu: 正規化線形関数 f(x) = max(0, x)
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        #sigmoid: 1 / 1 + exp(-x)
        #h = F.sigmoid(self.l1(x))
        #h = F.sigmoid(self.l2(h))

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


def imple(d, h1_unit, h2_unit, opt):
    
    train_data = d["train_data"]
    train_label = d["train_label"]
    test_data = d["test_data"]
    test_label = d["test_label"]

    model = ClassificationModel(h1_unit, h2_unit)
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
    print opt, h1_unit, h2_unit, "acc test ", acc.data
    model = optimizer = None


class MyThread(threading.Thread):
  def __init__(self, opts, d):
      threading.Thread.__init__(self)
      self.opts = opts
      self.d = d
                    
  def run(self):
      for opt in self.opts:  
          for h1_unit in [n * 100 for n in range(1, 10)] + [10, 50]:
          #for h1_unit in [100]:
              for h2_unit in [n * 100 for n in range(1, 10)] + [10, 50]:
              #for h2_unit in [50]:
                  imple(self.d, h1_unit, h2_unit, opt)
          


def nn(train_label, train_data, test_label, test_data):
    d = {}
    d["train_data"]= train_data.astype(np.float32)
    d["train_label"] = train_label.astype(np.int32)
    d["test_data"] = test_data.astype(np.float32)
    d["test_label"] = test_label.astype(np.int32)
    ao = ["SGD", "MomentumSGD", "AdaGrad", "AdaDelta", "Adam", "NesterovAG", "RMSprop", "RMSpropGraves", "SMORMS3"]
    #for opts in [ao[0:2], ao[2:4], ao[4:6], ao[6:7], ao[7:8], ao[8:]]:
    #for opt in ["AdaGrad"]:
    for opts in [ao[2:4] ]:
      t = MyThread(opts, d)
      t.start()
