#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-09-04 11:28
# @Author  : 冯佳欣
# @File    : util.py
# @Desc    : 一些辅助工具包
import os
import json
import numpy as np

def save_hparams(hparams, path):
    '''Saves hparams to path
    hparams: argsparse object.
    path: output directory.

    Writes
    hparams as literal dictionary to path.
    '''
    if not os.path.exists(path): os.makedirs(path)
    hp = json.dumps(vars(hparams))
    with open(os.path.join(path, "hparams"), 'w') as fout:
        fout.write(hp)

def shuffle_list(data):
    '''
    shuffle data
    :param data: [train_x,train_y]
    :return: data_list
    '''
    train_x = data[0]
    train_y = data[1]
    num_1 = train_x.shape[0]
    num_2 = train_y.shape[0]
    assert num_1 == num_2
    p = np.random.permutation(num_1)

    return [train_x[p],train_y[p]]

def batch_generator(data,batch_size,shuffle=True):
    '''
    yield batch_size data
    :param data: [x_train,y_train],其中x_train,y_train都是ndarray格式
    :param batch_size:
    :param shuffle:
    :return: yield
    '''
    train_x = data[0]
    train_y = data[1]
    num_1 = train_x.shape[0]
    num_2 = train_y.shape[0]
    assert num_1 == num_2
    num_length = num_1
    if shuffle:
        data = shuffle_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > num_length:
            batch_count = 0

            if shuffle:
                data = shuffle_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [train_x[start:end],train_y[start:end]]