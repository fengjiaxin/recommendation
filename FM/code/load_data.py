#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-09-03 20:15
# @Author  : 冯佳欣
# @File    : load_data.py
# @Desc    : 获取数据输入包

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# one hot encoder ，将label转换成one hot形式
def onehot_encoder(labels,num_classes):
    # labels 是Series的数据格式
    enc = LabelEncoder()
    labels = enc.fit_transform(labels)
    labels = labels.astype(np.int32)
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels,1)
    indices = tf.expand_dims(tf.range(0,batch_size,1),1)
    concated = tf.concat(axis=1,values=[indices,labels])
    onehot_labels = tf.sparse_to_dense(concated,tf.stack([batch_size,num_classes]),1.0,0.0)
    with tf.Session() as sess:
        return sess.run(onehot_labels)

# item数据
def load_item(item_path):
    item_columns = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children',
            'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
            'Thriller', 'War', 'Western']
    df_item = pd.read_csv(item_path, sep='|', names=item_columns,encoding="latin-1")
    df_item = df_item.drop(columns=['title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown'])
    return df_item


# user数据
def load_user(user_path):
    user_columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    df_user = pd.read_csv(user_path, sep='|', names=user_columns)

    # 给age分段
    df_user['age'] = pd.cut(df_user['age'], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                            labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90',
                                    '90-100'])
    # one hot 向量
    df_user = pd.get_dummies(df_user, columns=['gender', 'occupation', 'age'])
    df_user = df_user.drop(columns=['zip_code'])
    return df_user

# train数据
def load_train(train_path,item_path,user_path):
    df_item = load_item(item_path)
    df_user = load_user(user_path)
    train_columns = ['user_id', 'item_id', 'rating', 'timestamp']
    df_train = pd.read_csv(train_path, sep='\t', names=train_columns)

    # 将评分等于5的数据作为用户的点击数据，评分小于5分的数据作为用户的未点击数据，构造成一个而分类问题
    df_train['rating'] = df_train.rating.apply(lambda x: 1 if int(x) == 5 else 0)
    df_train = df_train.merge(df_user, on='user_id', how='left')
    df_train = df_train.merge(df_item, on='item_id', how='left')
    train_labels = onehot_encoder(df_train['rating'].astype(np.int32), 2)
    return df_train,train_labels

# test数据
def load_test(test_path,item_path,user_path):
    df_item = load_item(item_path)
    df_user = load_user(user_path)
    test_columns = ['user_id', 'item_id', 'rating', 'timestamp']
    df_test = pd.read_csv(test_path, sep='\t', names=test_columns)

    # 将评分等于5的数据作为用户的点击数据，评分小于5分的数据作为用户的未点击数据，构造成一个而分类问题
    df_test['rating'] = df_test.rating.apply(lambda x: 1 if int(x) == 5 else 0)
    df_test = df_test.merge(df_user, on='user_id', how='left')
    df_test = df_test.merge(df_item, on='item_id', how='left')
    test_labels = onehot_encoder(df_test['rating'].astype(np.int32), 2)
    return df_test, test_labels

def load_dataset(item_path,user_path,train_path,test_path):
    df_train,train_labels = load_train(train_path,item_path,user_path)
    df_test,test_labels = load_test(test_path,item_path,user_path)
    return df_train,train_labels,df_test,test_labels


if __name__ == '__main__':
    item_path = '../data/u.item'
    user_path = '../data/u.user'
    train_path = '../data/ua.base'
    test_path = '../data/ua.test'
    df_train, train_labels, df_test, test_labels = load_dataset(item_path,user_path,train_path,test_path)

    print('df_train:')
    print(df_train.head())
    print(train_labels[:5,:])

    print('################')
    print('df_test')
    print(df_test.head())
    print('test_labels:')
    print(test_labels[:5,:])
