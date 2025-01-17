import os
import math
import json
from progressbar import ProgressBar
import configparser
import numpy as np
from sklearn.metrics import mean_squared_error
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
            rmse = evaluate(model, sess, validation_data)
            epoch_data.append({'epoch': epoch, 'loss': total_loss, 'RMSE': rmse})
            if rmse < best_rmse:
                tf.train.Saver().save(sess, os.path.join(result_dir, 'model'))
            print('\n[Epoch {}] Loss = {:.2f}, RMSE = {:.4f}'.format(epoch, total_loss, rmse))
    return epoch_data


def get_feed_dict(model, train_data, start, end):
    feed_dict = {}
    feed_dict[model.feature_ids] = list(train_data[start:end, 1])
    feed_dict[model.label] = list(train_data[start:end, 0])
    return feed_dict


def evaluate(model, sess, evaluation_data):
    feed_dict = {model.feature_ids: list(evaluation_data[:, 1]), model.label: list(evaluation_data[:, 0])}
    predictions = np.reshape(model.predict(sess, feed_dict), (len(evaluation_data),))
    labels = np.reshape(list(evaluation_data[:, 0]), (len(evaluation_data),))
    predictions = np.maximum(predictions, [-1] * len(evaluation_data))
    predictions = np.minimum(predictions, [1] * len(evaluation_data))
    return math.sqrt(mean_squared_error(labels, predictions))


def save_train_result(result_dir, epoch_data):
    with open(os.path.join(result_dir, 'epoch_data.json'), 'w') as f:
        json.dump(epoch_data, f, indent=4)


def find_best_model(config, n_feature):
    best_model = None
    best_model_dir = None
    best_params = {}
    best_rmse = 100
    for batch_size in map(int, config['MODEL']['batch_size'].split()):
        for lr in map(float, config['MODEL']['lr'].split()):
            for l2_weight in map(float, config['MODEL']['l2_weight'].split()):
                for latent_dim in map(int, config['MODEL']['latent_dim'].split()):
                    result_dir = "data/train_result/batch_size_{}-lr_{}-l2_weight_{}-latent_dim_{}-epoch_{}".format(
                        batch_size, lr, l2_weight, latent_dim, config['MODEL']['epoch'])
                    with open(os.path.join(result_dir, 'epoch_data.json')) as f:
                        rmse = min([d['RMSE'] for d in json.load(f)])
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params = {'batch_size': batch_size, 'lr': lr, 'l2_weight': l2_weight, 'latent_dim': latent_dim}
                            tf.reset_default_graph()
                            best_model = FM(n_feature, lr, l2_weight, latent_dim)
                            best_model_dir = result_dir
    return best_model, best_model_dir, best_params


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
                    save_train_result(result_dir, epoch_data)

    best_model, best_model_dir, best_params = find_best_model(config, data_splitter.n_feature)
    with tf.Session() as sess:
        tf.train.Saver().restore(sess, os.path.join(best_model_dir, 'model'))
        rmse = evaluate(best_model, sess, test_data)
        print('---------------------------------\nBest result')
        print('batch_size = {}, lr = {}, l2_weight = {}, latent_dim = {}'.format(
            best_params['batch_size'], best_params['lr'], best_params['l2_weight'], best_params['latent_dim']))
        print('RMSE = {:.4f}'.format(rmse))


if __name__ == "__main__":
    main()
