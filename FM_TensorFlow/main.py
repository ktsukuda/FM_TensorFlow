import os
import math
from progressbar import ProgressBar
import configparser
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import data
from FM import FM


def train(result_dir, model, train_data, validation_data, batch_size, config):
    epoch_data = []
    best_rmse = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(config.getint('MODEL', 'epoch')):
            start = 0
            total_loss = 0
            np.random.shuffle(train_data)
            pb = ProgressBar(1, math.ceil(len(train_data)/batch_size))
            while start < len(train_data):
                _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + batch_size))
                start += batch_size
                total_loss += loss
                pb.update(start // batch_size)
            print('\n[Epoch {}] Loss = {:.2f}'.format(epoch, total_loss))
    return epoch_data


def get_feed_dict(model, train_data, start, end):
    feed_dict = {}
    feed_dict[model.feature_ids] = list(train_data[start:end, 1])
    feed_dict[model.label] = list(train_data[start:end, 0])
    return feed_dict


def main():
    config = configparser.ConfigParser()
    config.read('FM_TensorFlow/config.ini')

    data_splitter = data.DataSplitter()
    train_data = data_splitter.train
    validation_data = data_splitter.validation
    test_data = data_splitter.test

    for batch_size in map(int, config['MODEL']['batch_size'].split()):
        for lr in map(float, config['MODEL']['lr'].split()):
            for l2_weight in map(float, config['MODEL']['l2_weight'].split()):
                for latent_dim in map(int, config['MODEL']['latent_dim'].split()):
                    print('batch_size = {}, lr = {}, l2_weight = {}, latent_dim = {}'.format(
                        batch_size, lr, l2_weight, latent_dim))
                    result_dir = "data/train_result/batch_size_{}-lr_{}-l2_weight_{}-latent_dim_{}-epoch_{}".format(
                        batch_size, lr, l2_weight, latent_dim, config['MODEL']['epoch'])
                    os.makedirs(result_dir, exist_ok=True)
                    tf.reset_default_graph()
                    model = FM(data_splitter.n_feature, lr, l2_weight, latent_dim)
                    epoch_data = train(result_dir, model, train_data, validation_data, batch_size, config)


if __name__ == "__main__":
    main()
