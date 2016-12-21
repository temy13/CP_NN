#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np

def load_origin(filename):
    f = open('dataset/'+filename+'.txt', 'r')
    labels = []
    data = []
    for line in f:
        l = [float(item.replace(":", "").replace("\n", "")) for item in line.split(" ")]
        # labels.append(l[:1])
        # data.append(l[1:])
        labels.append(int(l[0]))
        data.append(l[1:])

    f.close()
    return labels, data


def save_json(save_list, key_name, indent=2):
    text = json.dumps(save_list, sort_keys=True, ensure_ascii=False, indent=indent)
    with open("jsondata/" + key_name + ".json", "w") as fh:
        if type(text) is unicode:
            fh.write(text.encode("utf-8"))
        else:
            fh.write(text)
    return


def load_json(key_name):
    with open("jsondata/" + key_name + ".json") as fh:
        read_dic = json.loads(fh.read(), "utf-8")
    return read_dic

def load_data():
    # train_label, train_data = load_origin("training")
    # test_label, test_data = load_origin("test")
    # save_json(train_label, "train_label", None)
    # save_json(train_data, "train_data", None)
    # save_json(test_label, "test_label", None)
    # save_json(test_data, "test_data", None)
    return \
        np.array(load_json("train_label")), \
        np.array(load_json("train_data")), \
        np.array(load_json("test_label")), \
        np.array(load_json("test_data"))
