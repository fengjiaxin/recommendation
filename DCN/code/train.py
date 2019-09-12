#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-09-11 20:07
# @Author  : 冯佳欣
# @File    : train.py
# @Desc    : train

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
from util import save_hparams,batch_norm_layer
from data_load import input_fn
from model import dcn_fn

def main(_):

    logging.info("# save hparams")
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()

    deep_layers_str = hp.deep_layer

    # 以模型layer结构作为子文件夹名称
    custom_model_path = deep_layers_str.replace(',','_')
    model_dir = os.path.join(hp.logdir,custom_model_path)


    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    save_hparams(hp, model_dir)


    model_params = {
        "field_size": hp.field_size,
        "feature_size": hp.feature_size,
        "embedding_size": hp.embedding_size,
        "learning_rate": hp.learning_rate,
        "l2_reg": hp.l2_reg,
        "deep_layer": hp.deep_layer,
        "dropout": hp.dropout,
        "cross_layer":hp.cross_layer,
        "batch_norm": hp.batch_norm,
        "optimizer": hp.optimizer
    }

    config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0, 'CPU': 10}),
        log_step_count_steps=hp.log_steps, save_summary_steps=hp.log_steps)


    logging.info('initialize DCN model')
    dcn = tf.estimator.Estimator(model_fn=dcn_fn, model_dir=model_dir, params=model_params, config=config)


    if hp.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(hp.train_path, num_epochs=hp.num_epochs, batch_size=hp.batch_size))
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(hp.valid_path, num_epochs=1, batch_size=hp.batch_size), steps=None, start_delay_secs=1000, throttle_secs=1200)
        tf.estimator.train_and_evaluate(dcn, train_spec, eval_spec)
    elif hp.task_type == 'eval':
        dcn.evaluate(input_fn=lambda: input_fn(hp.valid_path, num_epochs=1, batch_size=hp.batch_size))
    elif hp.task_type == 'infer':
        preds = hp.predict(input_fn=lambda: input_fn(hp.test_path, num_epochs=1, batch_size=hp.batch_size), predict_keys="probalities")
        with open(hp.data_dir+"/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()