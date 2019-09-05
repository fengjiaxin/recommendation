#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-09-04 10:31
# @Author  : 冯佳欣
# @File    : train.py
# @Desc    : 训练过程

import os
import tensorflow as tf
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
import numpy as np
from hparams import Hparams
from util import save_hparams,batch_generator
from load_data import load_train,get_feature2field_dict
from model import FFM
os.environ['KMP_DUPLICATE_LIB_OK']='True'


logging.info("# save hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()

if not os.path.exists(hp.logdir):
    os.makedirs(hp.logdir)
save_hparams(hp, hp.logdir)


logging.info("# Prepare train data")
df_train,train_labels = load_train(hp.train_path,hp.item_path,hp.user_path,hp.prefix_sep)
logging.info('df_train.shape ' + str(df_train.shape))
logging.info('train_labels.shape ' + str(train_labels.shape))

# 特征长度
feature_length = df_train.shape[1]
hp.feature_length = feature_length

# 特征名称列表
feature_cols = df_train.columns.tolist()
# 获取feature2field_dict
feature2field_dict,field_list = get_feature2field_dict(feature_cols,hp.prefix_sep)
hp.field_num = len(field_list)

# 样本数量
train_num = df_train.shape[0]

# 数据生成器
batch_gen = batch_generator([df_train.values,train_labels],hp.batch_size)


# initialize FFM model
logging.info('initialize FFM model')
fm_model = FFM(hp,feature2field_dict)
fm_model.build_graph()


# begin session
logging.info('# Session')
saver = tf.train.Saver(max_to_keep=hp.max_to_keep)
with tf.Session() as sess:
    # 恢复数据
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        logging.info('initialize fresh parameters for the ffm model')
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess,ckpt)

    # merga all the summaries and write them out to train_logs
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(hp.logdir,sess.graph)

    # get num_batches
    num_batches = train_num//hp.batch_size + 1

    for e in range(hp.num_epochs):
        num_samples = 0
        losses = []
        for ibatch in range(num_batches):
            # batch_size data
            batch_x,batch_y = next(batch_gen)
            batch_y = np.array(batch_y).astype(np.float32)
            actual_batch_size = len(batch_y)

            # create a feed dic
            feed_dict = {fm_model.xs:batch_x,
                        fm_model.ys:batch_y}

            loss,acc,summary,global_step,_ = sess.run([fm_model.loss,fm_model.accuracy,
                                                       merged,fm_model.global_step,
                                                       fm_model.train_op],feed_dict=feed_dict)

            losses.append(loss * actual_batch_size)
            num_samples += actual_batch_size

            # record summaries and accuracy
            summary_writer.add_summary(summary,global_step=global_step)

            # print train_loss
            if global_step % hp.print_every == 0:
                logging.info("Iterarion {0} : with minibatch training loss = {1} and accuracy of {2}".format(global_step,loss,acc))

                model_output = 'train_E%dL%.2f'%(e,loss)

                logging.info("# save models")
                ckpt_name = os.path.join(hp.logdir, model_output)
                saver.save(sess, ckpt_name, global_step=global_step)
                logging.info("after training of {} epochs, {} has been saved.".format(e, ckpt_name))

        # print loss of one epoch
        total_loss = np.sum(losses)/num_samples
        logging.info('Epoch {1}, Overall loss = {0:.3g}'.format(total_loss,e+1))
logging.info('##############train done##############')
