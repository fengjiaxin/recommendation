#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-09-11 15:14
# @Author  : 冯佳欣
# @File    : model.py
# @Desc    : DCN模型

import tensorflow as tf
from util import batch_norm_layer
import logging

def dcn_fn(features,labels,mode,params):
    logging.info('### 1. begin read hp parameters')
    field_size = params["field_size"]
    feature_size = params["feature_size"]
    # 嵌入向量个数
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    # layers list
    deep_layers_list = [int(layer) for layer in params['deep_layer'].split(',')]
    # drop list
    dropout_list = [float(layer) for layer in params['dropout'].split(',')]
    # cross layers数
    cross_layer = params["cross_layer"]
    # batch_norm
    batch_norm = params["batch_norm"]
    #batch_norm_decay = hp.batch_norm_decay
    #loss_type = hp.loss_type
    optimizer_type = params["optimizer"]


    logging.info('### 2. build weights')
    # ---- build weights,每个类别都映射为embedding_size向量
    Cross_B = tf.get_variable(name='cross_b',shape=[cross_layer,field_size * embedding_size],initializer=tf.glorot_normal_initializer())
    Cross_W = tf.get_variable(name='cross_w',shape=[cross_layer,field_size * embedding_size],initializer=tf.glorot_normal_initializer())
    Feat_Emb = tf.get_variable(name='emb',shape=[feature_size,embedding_size],initializer=tf.glorot_normal_initializer())

    logging.info('### 3. get features')
    # ---- get features
    feat_ids = features['feat_ids']
    feat_ids = tf.reshape(feat_ids,shape=[-1,field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals,shape=[-1,field_size])

    logging.info('### 4. get X0')
    # ---- get X0
    with tf.variable_scope('Embedding_layer'):
        # [None,field_size,embedding_size]
        embeddings = tf.nn.embedding_lookup(Feat_Emb,feat_ids)
        # [None,field_size,1]
        feat_vals = tf.reshape(feat_vals,shape=[-1,field_size,1])
        # 将向量和值相乘，最后一个维度不同，相乘相当于一个标量乘以一个向量
        embeddings = tf.multiply(embeddings,feat_vals)
        # [None,field_size * embedding_size]
        x0 = tf.reshape(embeddings,shape=[-1,field_size * embedding_size])

    logging.info('### 5. cross network')
    # ---- cross network
    with tf.variable_scope('cross-network'):
        # 计算公式 𝑥𝑙+1=𝑥0𝑥𝑇𝑙𝑤𝑙+𝑏𝑙+𝑥𝑙
        xl = x0
        for l in range(cross_layer):
            # [field_size * embedding_size,1]
            wl = tf.reshape(Cross_W[l],shape=[-1,1])
            # xlw是一个标量
            xlw = tf.matmul(xl,wl)
            xl = tf.multiply(x0,xlw) + xl + Cross_B[l]

    logging.info('### 6. deep network')
    # ---- deep network
    with tf.variable_scope('deep-network'):
        if batch_norm:
            if mode == 'train':
                train_phase = True
            else:
                train_phase = False

        # deep input:[None,field_size * embedding_size]
        x_deep = x0
        for i in range(len(deep_layers_list)):
            x_deep = tf.contrib.layers.fully_connected(inputs=x_deep,num_outputs=deep_layers_list[i],
                                                      weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),scope='mlp%d'%i)
            if batch_norm:
                x_deep = batch_norm_layer(x_deep,train_phase=train_phase,scope_bn='bn_%d'%i)
            if mode == tf.estimator.ModeKeys.TRAIN:
                x_deep = tf.nn.dropout(x_deep,keep_prob=dropout_list[i])

    logging.info('### 7 dcn out')
    # ---- dcn out
    with tf.variable_scope('dcn-out'):
        # [None,field_size * embedding_size + deep_layer[-1]]
        x_stack = tf.concat([xl,x_deep],axis=1)
        y = tf.contrib.layers.fully_connected(inputs=x_stack, num_outputs=1, activation_fn=tf.identity, weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='out_layer')
        logits = tf.reshape(y,shape=[-1])

        # 分类数据
        predict_classes = tf.sign(logits)
        probalities = tf.nn.sigmoid(logits)



    predictions = {'class_ids':predict_classes,
                   'probalities':probalities,
                   'logits':logits}

    ######### predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)

    # bulld loss
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=probalities)
    # 正则项损失
    regular_loss = l2_reg * tf.nn.l2_loss(Cross_W) + l2_reg * tf.nn.l2_loss(
        Cross_W) + l2_reg * tf.nn.l2_loss(Feat_Emb)
    loss = tf.reduce_mean(cross_entropy + regular_loss)

    ######### eval
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predict_classes,
                                   name='acc_op')
    auc = tf.metrics.auc(labels=labels,
                         predictions=probalities,
                         name='acu_op')

    eval_metric_ops = {
        "auc": auc,
        'accuracy':accuracy
    }
    tf.summary.scalar('accuracy',accuracy[1])
    tf.summary.scalar('acu', auc[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

    # ------bulid optimizer------
    if optimizer_type == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif optimizer_type == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif optimizer_type == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif optimizer_type == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=probalities,
            loss=loss,
            train_op=train_op)
