#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-09-11 19:37
# @Author  : 冯佳欣
# @File    : data_load.py
# @Desc    : 读取数据辅助工具包

import tensorflow as tf
import logging

def input_fn(filenames,batch_size=256,num_epochs=100,perform_shuffle=True,buffer_size=256,prefetch_size=1000):
    logging.info('Parsing %s'%filenames)
    def decode_libsvm(line):
        columns = tf.string_split([line],' ')
        labels = tf.string_to_number(columns.values[0],out_type=tf.float32)
        splits = tf.string_split(columns.values[1:],':')
        id_vals = tf.reshape(splits.values,splits.dense_shape)
        feat_ids,feat_vals = tf.split(id_vals,num_or_size_splits=2,axis=1)
        feat_ids = tf.string_to_number(feat_ids,out_type=tf.int32)
        feat_vals = tf.string_to_number(feat_vals,out_type=tf.float32)
        return {"feat_ids": feat_ids, "feat_vals": feat_vals},labels

    # extract lines from input files use the datast api
    dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm,num_parallel_calls=10).prefetch(prefetch_size)

    # randomizes input using a window
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # epochs
    dataset = dataset.repeat(num_epochs)
    # batch
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_features,batch_labels = iterator.get_next()
    return batch_features,batch_labels

