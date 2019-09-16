#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-09-16 17:35
# @Author  : 冯佳欣
# @File    : train.py
# @Desc    : train
import tensorflow as tf
from util_helper import load_data,generate_test,generate_train_batch,generate_test_batch,bmp_mf

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
# constant

num_epochs=10
num_batchs=500
latent_num=20
print_every=100


logging.info('# get user_count,item_count,user_ratings')
data_path = '../data/ml-100k/u.data'
user_count,item_count,user_ratings = load_data(data_path)
user_ratings_test = generate_test(user_ratings)

logging.info('# Session')
with tf.Session() as sess:
    u,i,j,mf_auc,bpr_loss,train_op = bmp_mf(user_count,item_count,latent_num)
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        _batch_bprloss = 0
        for k in range(num_batchs):
            u_ij = generate_train_batch(user_ratings,user_ratings_test,item_count)
            _bpr_loss,_train_op = sess.run([bpr_loss,train_op],
                                          feed_dict={u:u_ij[:,0],i:u_ij[:,1],j:u_ij[:,2]})
            _batch_bprloss += _bpr_loss

            global_step = epoch * num_epochs + k + 1
        logging.info('######## epochs: %d'%(epoch))
        logging.info('######## bpr_loss: %.4f'%(_batch_bprloss/num_batchs))

    user_count = 0
    _auc_sum = 0.0
    _bpr_loss_sum = 0.0

    for t_uij in generate_test_batch(user_ratings,user_ratings_test,item_count):
        _auc,_test_bpr_loss = sess.run([mf_auc,bpr_loss],
                                       feed_dict={u:t_uij[:,0],i:t_uij[:,1],j:t_uij[:,2]})
        user_count += 1
        _auc_sum += _auc
        _bpr_loss_sum += _test_bpr_loss
    logging.info('test_loss: %.4f\ttest_auc:%.4f'%(_bpr_loss_sum/user_count,_auc_sum/user_count))