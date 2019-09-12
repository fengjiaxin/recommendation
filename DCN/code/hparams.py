#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-09-11 20:07
# @Author  : 冯佳欣
# @File    : hparams.py
# @Desc    : 超参数

import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # input_path
    parser.add_argument('--train_path',default = '../data/tr.mini.libsvm',help="train data path")
    parser.add_argument('--valid_path', default='../data/va.mini.libsvm', help="valid data path")
    parser.add_argument('--test_path', default='../data/te.mini.libsvm', help="test data path")


    # model parameters
    parser.add_argument('--field_size', default=39, type=int,help="number of fields")
    parser.add_argument('--feature_size', default=117581, type=int,help="number of features")
    parser.add_argument('--embedding_size', default=64, type=int,help="embedding size")
    parser.add_argument('--learning_rate', default=0.01, type=float,help="learning rate")
    parser.add_argument('--batch_size', default=64, type=int,help="number of batch size")
    parser.add_argument('--num_epochs', default=10, type=int,help="number of epochs")
    parser.add_argument('--l2_reg', default=0.0001, type=float,help="l2 regularization")
    parser.add_argument('--log_steps', default=500, type=int,help="save summary every steps")
    #parser.add_argument('--loss_type', default='log_loss', help='loss type{square_loss,log_loss}', type=str)
    parser.add_argument('--optimizer', default='Adam', help='optimizer type{Adam,Adagrad,Momentum,Ftrl}', type=str)
    parser.add_argument('--deep_layer', default='512,256,128', help='deep layers str,sep by ,', type=str)
    parser.add_argument('--dropout', default='0.5,0.5,0.5', help='drop str,sep by ,', type=str)
    parser.add_argument('--cross_layer', default=2, help="cross network layer num",type=int)
    parser.add_argument('--batch_norm', default=False,help="perform batch nmormaization {True,False}",type=bool)
    #parser.add_argument('--batch_norm_decay', default=0.9, type=float, help="decay for the moving average(recommend trying decay=0.9)")

    # mode
    parser.add_argument('--task_type', default = 'train',help='task type {train, infer, eval}', type=str)

    # save_path
    parser.add_argument('--logdir', default="../model", help="model check point dir")
    parser.add_argument('--datadir', default="../data", help="model check point dir")

