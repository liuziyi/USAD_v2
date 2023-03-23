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

from imblearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import ClusterCentroids,RandomUnderSampler,TomekLinks


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


def sampling_data_prep(method, X_train, y_train):
    print("\n本次采样方法：{}".format(method))
    if (method == "tomeklink"):
        undersample = TomekLinks()
        x, y = undersample.fit_resample(X_train, y_train)

    elif (method == "smote"):
        oversample = SMOTE()
        x, y = oversample.fit_resample(X_train, y_train)

    elif (method == "adasyn"):
        oversample = ADASYN()
        x, y = oversample.fit_resample(X_train, y_train)

    elif (method == 'randomover'):
        oversample = RandomOverSampler(random_state=0)
        x, y = oversample.fit_resample(X_train, y_train)

    elif(method == 'nonsampling'):
        x, y = X_train, y_train

    elif (method == 'combine_over_and_under_sampling'):
        over = SMOTE()
        under = TomekLinks()
        # define pipeline
        pipeline = Pipeline(steps=[('o', over), ('u', under)])
        x, y = pipeline.fit_resample(X_train, y_train)

    count_fraud = 0
    count_nonfraud = 0
    for flag in y:
        if flag == 0:
            count_fraud += 1
        else:
            count_nonfraud += 1
    print("number of fraud: {}, number of non fraud: {}".format(count_fraud, count_nonfraud))

    return x, y


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
    X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    total_fit = 0
    total_predict = 0

    entities = 1
    
    #采样方式名称
    sample_name = ['tomeklink', 'smote', 'adasyn', 'randomover', 'combine_over_and_under_sampling', 'nonsampling']
    i = 0

    X_train, x_test = preprocess(np.asarray(X_train), np.asarray(X_test))
    
    #每种采样方式的分数存在这些列表里
    sampling_train_score = []
    sampling_test_score = []
    sampling_m = []
    sampling_threshold = []
    
    
    #最终训练集x_train, y_train,测试集x_test, y_test 都是小写
    for name in sample_name:
        
        x_train, y_train = sampling_data_prep(name, X_train, Y_train)

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
        sampling_train_score.append(train_score)

        if not os.path.exists(config.result_dir + f'/{i}'):
            os.mkdir(config.result_dir + f'/{i}')

        if config.train_score_filename:
            with open(os.path.join(config.result_dir, f'{i}/{config.train_score_filename}'), 'wb') as file:
                pickle.dump(train_score, file)

        # get test score
        start_predict = time.time()
        test_score = model.predict(x_test)
        sampling_test_score.append(test_score)
        end_predict = time.time()
        total_predict += (end_predict - start_predict)
        if config.test_score_filename:
            with open(os.path.join(config.result_dir, f'{i}/{config.test_score_filename}'), 'wb') as file:
                pickle.dump(test_score, file)

        # m = [f1, precision, recall, TP, TN, FP, FN, latency]
        # 超过threshold的为异常，低于的为正常。待用m和threshold画图
        m, threshold = bf_search(test_score, y_test, 0, 1, 10000, 2000)
        with open(os.path.join(config.result_dir, f'{i}/{config.threshold_filename}'), 'w') as file:
            file.write(str(threshold))
        with open(os.path.join(config.result_dir, f'{i}/score'), 'w') as file:
            file.write(str(m))
            
        sampling_m.append(m)
        sampling_threshold.append(threshold)
    
    i+=1
    print(f'\n总训练耗时: {total_fit}, 总预测耗时: {total_predict}')
   
    j = 0
    print("\n")
    for name in sample_name:

        print("USAD with {} sample method:".format(name))
        print("train score: {}".format(sampling_train_score[j]))
        print("test score: {}".format(sampling_test_score[j]))
        print("f1: {}, precision: {}, recall: {}, TP: {}, TN: {}, FP: {}, FN: {}, latency: {}".format(sampling_m[j][0],sampling_m[j][1],sampling_m[j][2],sampling_m[j][3],sampling_m[j][4],sampling_m[j][5],sampling_m[j][6],sampling_m[j][7]))
        print("best threshold: {}\n".format(sampling_threshold[j]))
        j += 1


if __name__ == '__main__':
    print(config)
    main()
