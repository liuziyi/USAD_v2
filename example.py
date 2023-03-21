import argparse
import os
import pickle
import logging
import numpy as np
import time
import pandas as pd

import torch

from usad.model_v2 import USAD
from usad.evaluate import bf_search
from usad.utils import get_data, ConfigHandler, merge_data_to_csv, get_threshold
from sklearn.preprocessing import MinMaxScaler
import time, random, json
from sklearn.model_selection import train_test_split


# normalize trian and test data
def preprocess(df_train, df_test):
    """
    normalize raw data
    """
    df_train = np.asarray(df_train, dtype=np.float32)
    df_test = np.asarray(df_test, dtype=np.float32)
    if len(df_train.shape) == 1 or len(df_test.shape) == 1:
        raise ValueError('Data must be a 2-D array')
    if np.any(sum(np.isnan(df_train)) != 0):
        print('train data contains null values. Will be replaced with 0')
        df_train = np.nan_to_num(df_train)
    if np.any(sum(np.isnan(df_test)) != 0):
        print('test data contains null values. Will be replaced with 0')
        df_test = np.nan_to_num(df_test)
    scaler = MinMaxScaler()
    scaler = scaler.fit(df_train)
    df_train = scaler.transform(df_train)
    df_test = scaler.transform(df_test)
    return df_train, df_test


def main():
    """# Import data"""

    data = pd.read_csv('data/creditcard.csv')
    data["Time"] = data["Time"].apply(lambda x: x / 3600 % 24)

    data.head()

    a = data.Class.value_counts()

    """# Preparation"""

    nf = data[data['Class'] == 0].sample(492 * 9, replace=True)
    f = data[data['Class'] == 1]
    df = nf.append(f).sample(frac=1).reset_index(drop=True)
    X = df.drop(['Class'], axis=1).values
    Y = df["Class"].values

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    total_fit = 0
    total_predict = 0

    entities = 1

    for i in range(entities):

        x_train, x_test = preprocess(np.asarray(X_train), np.asarray(X_test))

        # init model
        model = USAD(x_dims=config.x_dims[config.dataset], max_epochs=config.max_epochs[config.dataset],
                     batch_size=config.batch_size, z_dims=config.z_dims,
                     window_size=config.window_size[config.dataset],
                     valid_step_frep=config.valid_step_freq)

        # restore model
        if config.restore_dir:
            print(f'Restore model from `{config.restore_dir}`')
            shared_encoder_path = os.path.join(config.restore_dir, 'shared_encoder.pkl')
            decoder_G_path = os.path.join(config.restore_dir, 'decoder_G.pkl')
            decoder_D_path = os.path.join(config.restore_dir, 'decoder_D.pkl')
            model.restore(shared_encoder_path, decoder_G_path, decoder_D_path)
        # train model
        else:
            start_fit = time.time()
            model.fit(x_train)
            end_fit = time.time()
            total_fit += (end_fit - start_fit)


        # save model
        if config.save_dir:
            if not os.path.exists(config.save_dir + f'/{i}'):
                os.mkdir(config.save_dir + f'/{i}')
            shared_encoder_path = os.path.join(config.save_dir, f'{i}/shared_encoder.pkl')
            decoder_G_path = os.path.join(config.save_dir, f'{i}/decoder_G.pkl')
            decoder_D_path = os.path.join(config.save_dir, f'{i}/decoder_D.pkl')
            model.save(shared_encoder_path, decoder_G_path, decoder_D_path)

        # get train score
        train_score = model.predict(x_train)

        if not os.path.exists(config.result_dir + f'/{i}'):
            os.mkdir(config.result_dir + f'/{i}')

        if config.train_score_filename:
            with open(os.path.join(config.result_dir, f'{i}/{config.train_score_filename}'), 'wb') as file:
                pickle.dump(train_score, file)

        # get test score
        start_predict = time.time()
        test_score = model.predict(x_test)
        end_predict = time.time()
        total_predict += (end_predict - start_predict)
        if config.test_score_filename:
            with open(os.path.join(config.result_dir, f'{i}/{config.test_score_filename}'), 'wb') as file:
                pickle.dump(test_score, file)

        # get threshold
        # 超过threshold的为异常，低于的为正常。用y_test和test_score计算分数并画图
        threshold = get_threshold(y_test, test_score)
        with open(os.path.join(config.result_dir, f'{i}/{config.threshold_filename}'), 'w') as file:
            file.write(str(threshold))

    print(f'总训练耗时: {total_fit}, 总预测耗时: {total_predict}')


if __name__ == '__main__':
    # get config
    config = ConfigHandler().config
    print('Configuration:')
    print(config)
    main()
