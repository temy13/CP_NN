#!/usr/bin/env python
# -*- coding: utf-8 -*-
from helper import *
from nn import nn


if __name__ == '__main__':
    print ("loading...")
    train_label, train_data, test_label, test_data = load_data()
    # print len(train_label), len(train_data), len(test_label), len(test_data)
    # print len(train_label[0]), len(train_data[0]), len(test_label[0]), len(test_data[0])
    print ("training...")
    nn(train_label, train_data, test_label, test_data)
