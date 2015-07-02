# -*- coding:utf-8 -*-
from mytoolbox import *
import cPickle as pickle
import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import Image
import os
from easydict import EasyDict as edict
from time import time

P               = edict({})
P.data_dir      = './data/'
P.cache_dir     = './cache/'
P.model_dir     = './models/'
P.result_dir    = './result/'
P.model_name    = ''
P.gpu           = 0
P.max_width     = 540
P.max_height    = 420

P.reduced = 3

def import_data():
    if os.path.exists(os.path.join(P.cache_dir, 'x_test.npy')):
        x_test    = np.load(os.path.join(P.cache_dir, 'x_test.npy'))
        s_test    = np.load(os.path.join(P.cache_dir, 's_test.npy'))
        name_test = pickleload(os.path.join(P.cache_dir, 'name_test.p'))
        return x_test, s_test, name_test
    test_list = os.listdir(os.path.join(P.data_dir, 'test'))
    x_test = np.zeros((len(test_list), 1, P.max_width+2*P.reduced, P.max_height+2*P.reduced))
    s_test = np.zeros((len(test_list), 2))
    name_test = []
    for count, i in enumerate(test_list):
        input_image = np.array(Image.open(os.path.join(P.data_dir, 'test', i)))
        input_image = 1 - input_image.astype(np.float32).T / 255
        s_test[count, 0] = input_image.shape[0]
        s_test[count, 1] = input_image.shape[1]
        x_test[count:count+1, 0:1, P.reduced:s_test[count, 0]+P.reduced, P.reduced:s_test[count, 1]+P.reduced] = input_image
        name_test.append(i)
    np.save(os.path.join(P.cache_dir, 'x_test'), x_test)
    np.save(os.path.join(P.cache_dir, 's_test'), s_test)
    pickledump(os.path.join(P.cache_dir, 'name_test.p'), name_test)
    return x_test, s_test, name_test

def save_as_image(mat, name):
    assert len(y.shape) == 2
    img = Image.fromarray(mat)
    img.save(name)

def test_and_save(model):
    x_test, s_test, name_test = import_data()
    y_test = np.zeros((len(name_test), P.max_height, P.max_width))
    for i, name in enumerate(name_test):
        y = forward(x_test[i:i+1, 0:1, 0:s_test[i, 0]+2*P.reduced, 0:s_test[i, 1]+2*P.reduced], model)
        if P.gpu >= 0:
            y = cuda.to_cpu(y.data)
        else:
            y = y.data
        assert y.shape[0] == s_test[i, 0] and y.shape[1] == s_test[i, 1]
        y = 1 - s_test[0, 0, s_test[i, 0], s_test[i, 1]].T
        y_test[i, 0:s_test[i, 1], 0:s_test[i, 0]] = y
        save_as_mat(y, os.path.join(P.result_dir, name))
        # csvファイルとして保存する機能は未実装

def forward(x_data, model):
    x = Variable(x_data)
    h = F.relu(model.conv1(x))
    h = F.relu(model.conv2(h))
    h = F.relu(model.conv3(h))
    return h

def main():
    if P.gpu >= 0:
        cuda.init(P.gpu)
    model = pickle.load(open(os.path.join(P.model_dir, P.model_name), 'rb'))
    test_and_save(model)
    return

if __name__ == '__main__':
    main()
