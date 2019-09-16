#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-09-16 17:34
# @Author  : 冯佳欣
# @File    : util_helper.py
# @Desc    : 辅助工具

import tensorflow as tf
import numpy as np
import os
import random
from collections import defaultdict


def load_data(data_path):
    '''

    returns:
    max_user_id:最大user_id
    max_item_id:最大item_id
    user_ratings:每个用户看过的电影都存储在user_ratings，key是user_id，value是item id的set集合
    '''
    user_ratings = defaultdict(set)
    max_user_id = 0
    max_item_id = 0
    with open(data_path, 'r') as f:
        for line in f:
            vec = line.strip().split('\t')
            user_id = int(vec[0])
            item_id = int(vec[1])
            user_ratings[user_id].add(item_id)
            max_user_id = max(max_user_id, user_id)
            max_item_id = max(max_item_id, item_id)

    print('max_user_id:%d' % (max_user_id))
    print('max_item_id:%d' % (max_item_id))
    return max_user_id, max_item_id, user_ratings

def generate_test(user_ratings):
    '''
    构建测试数据集合，对每一个用户，在user_ratings中随机找到他评分过的一部电影i，保存在user_ratings_test中
    '''
    user_test = dict()
    for u,i_list in user_ratings.items():
        user_test[u] = random.sample(user_ratings[u],1)[0]
    return user_test

def generate_train_batch(user_ratings,user_ratings_test,item_count,batch_size=512):
    '''
    构建若干批训练数据集，主要是根据user_ratings找到若干训练用的三元组<u,i,j>,对于随机抽取出的用户u,i可以从user_ratings中水机抽取
    ，而j也是从总的电影集合中随机抽取，当然保证(u,j)不出现在user_ratings中。
    '''
    t = []
    for b in range(batch_size):
        u = random.sample(user_ratings.keys(),1)[0]
        i = random.sample(user_ratings[u],1)[0]
        # 保证选取的i和测试集中的数据不重复
        while i == user_ratings_test[u]:
            i = random.sample(user_ratings[u],1)[0]
        # 随机挑选j
        j = random.randint(1,item_count)
        while j in user_ratings[u]:
            j = random.randint(1,item_count)
        t.append([u,i,j])
    return np.asarray(t)

def generate_test_batch(user_ratings,user_ratings_test,item_count):
    for u in user_ratings.keys():
        t = []
        i = user_ratings_test[u]
        for j in range(1,item_count+1):
            if j not in user_ratings[u]:
                t.append([u,i,j])
        yield np.asarray(t)

def bmp_mf(user_count, item_count, hidden_dim):
    u = tf.placeholder(tf.int32, [None])
    i = tf.placeholder(tf.int32, [None])
    j = tf.placeholder(tf.int32, [None])

    user_embed_w = tf.get_variable('user_embed_w', [user_count + 1, hidden_dim],
                                   initializer=tf.random_normal_initializer(0, 0.1))
    item_embed_w = tf.get_variable('item_embed_w', [item_count + 1, hidden_dim],
                                   initializer=tf.random_normal_initializer(0, 0.1))

    # 查询embedding向量 [None,hidden_dim]
    u_emb = tf.nn.embedding_lookup(user_embed_w, u)
    i_emb = tf.nn.embedding_lookup(item_embed_w, i)
    j_emb = tf.nn.embedding_lookup(item_embed_w, j)

    # mf predict u_i > u_j
    x = tf.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1, keep_dims=True)

    # auc
    mf_auc = tf.reduce_mean(tf.to_float(x > 0))

    l2_norm = tf.add_n([
        tf.reduce_sum(tf.multiply(u_emb, u_emb)),
        tf.reduce_sum(tf.multiply(i_emb, i_emb)),
        tf.reduce_sum(tf.multiply(j_emb, j_emb))
    ])

    regulation_rate = 0.001
    bpr_loss = regulation_rate * l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(x)))

    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(bpr_loss)

    return u, i, j, mf_auc, bpr_loss, train_op