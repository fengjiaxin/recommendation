#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-09-04 10:33
# @Author  : 冯佳欣
# @File    : hparams.py
# @Desc    : 超参数设置

import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # input_path

    parser.add_argument('--item_path',default = '../data/u.item',help="item path")
    parser.add_argument('--user_path', default='../data/u.user', help="user path")
    parser.add_argument('--train_path', default='../data/ua.base', help="train path")
    parser.add_argument('--test_path', default='../data/ua.test', help="test path")


    # model parameters
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--feature_length', default=20, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--k', default=40, type=int)
    parser.add_argument('--reg_l1', default=2e-2, type=float)
    parser.add_argument('--reg_l2', default=0, type=int)

    # mode
    parser.add_argument('--mode', default = 'train',help='train or test', type=str)

    # save_path
    parser.add_argument('--logdir', default="../logs", help="log directory")
    parser.add_argument('--max_to_keep', default=5, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--print_every', default=100, type=int)
