#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-09-03 16:44
# @Author  : 冯佳欣
# @File    : model.py
# @Desc    : FM模型的类

import tensorflow as tf

class FM(object):
    def __init__(self,hp):
        self.num_classes = hp.num_classes
        # 缩减为k维向量
        self.k = hp.k
        self.lr = hp.lr
        self.batch_size = hp.batch_size
        self.feature_length = hp.feature_length
        self.reg_l1 = hp.reg_l1
        self.reg_l2 = hp.reg_l2

    def add_input(self):
        self.xs = tf.placeholder('float32', [None, self.feature_length])
        self.ys = tf.placeholder('float32', [None, self.num_classes])



    # 前向传递过程
    def inference(self):
        '''
        前向传播过程
        '''
        # 线性层
        with tf.variable_scope('linera_layer'):
            # 常数项
            w0 = tf.get_variable('w0',shape=[self.num_classes],
                                 initializer=tf.zeros_initializer())

            # 线性特征
            self.w = tf.get_variable('w',shape=[self.feature_length,self.num_classes],
                                     initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01))
            # 线性特征加常数项
            self.linear_terms = tf.add(tf.matmul(self.xs,self.w),w0)

        # 交叉特征层
        with tf.variable_scope('interaction_layer'):
            v = tf.get_variable('v',shape=[self.feature_length,self.k],
                                     initializer = tf.truncated_normal_initializer(mean=0,stddev=0.01))

            # [None,k]
            a = tf.pow(tf.matmul(self.xs,v),2)
            # [None,k]
            b = tf.matmul(tf.pow(self.xs,2),tf.pow(v,2))

            self.interaction_terms = tf.multiply(0.5,tf.reduce_sum(tf.subtract(a,b),1,keepdims=True))

        # 输出
            self.y_out = tf.add(self.linear_terms,self.interaction_terms)

        if self.num_classes == 2:
            self.y_out_prob = tf.nn.sigmoid(self.y_out)
        elif self.num_classes > 2:
            self.y_out_prob = tf.nn.softmax(self.y_out)

    def add_loss(self):
        if self.num_classes == 2:
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.ys, logits=self.y_out)
        elif self.num_classes > 2:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.ys, logits=self.y_out)
        mean_loss = tf.reduce_mean(cross_entropy)
        self.loss = mean_loss
        tf.summary.scalar('loss', self.loss)

    def add_accuracy(self):
        # accuracy
        self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.y_out, 1), tf.float32),
                                           tf.cast(tf.argmax(self.ys, 1), tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        # add summary to accuracy
        tf.summary.scalar('accuracy', self.accuracy)

    def train(self):
        self.global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.FtrlOptimizer(self.lr, l1_regularization_strength=self.reg_l1,
                                           l2_regularization_strength=self.reg_l2)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def build_graph(self):
        self.add_input()
        self.inference()
        self.add_loss()
        self.add_accuracy()
        self.train()