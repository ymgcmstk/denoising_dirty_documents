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
import gzip

P               = edict({})
P.data_dir      = './data/'
P.cache_dir     = './cache/'
P.model_dir     = './models/'
P.result_dir    = './result/'
P.model_name    = 'model_0d266390153207.cPickle'
P.submission    = 'submission.txt.gz'
P.gpu           = 0
P.max_width     = 540
P.max_height    = 420

P.again = False

P.reduced = 6

def import_data():
    if P.again:
        test_list = os.listdir(P.result_dir)
    else:
        test_list = os.listdir(os.path.join(P.data_dir, 'test'))
    x_test = np.zeros((len(test_list), 1, P.max_width+2*P.reduced, P.max_height+2*P.reduced))
    s_test = np.zeros((len(test_list), 2))
    name_test = []
    count = 0
    for i in test_list:
        if not '.png' in i:
            continue
        if P.again:
            input_image = np.array(Image.open(os.path.join(P.result_dir, i)))
        else:
            input_image = np.array(Image.open(os.path.join(P.data_dir, 'test', i)))
        input_image = 1 - input_image.astype(np.float32).T / 255
        s_test[count, 0] = input_image.shape[0]
        s_test[count, 1] = input_image.shape[1]
        x_test[count:count+1, 0:1, P.reduced:s_test[count, 0]+P.reduced, P.reduced:s_test[count, 1]+P.reduced] = input_image
        name_test.append(i)
        count += 1
    return x_test, s_test, name_test

def save_as_image(mat, name):
    assert len(mat.shape) == 2
    mat = np.uint8(mat * 255)
    img = Image.fromarray(mat)
    img.save(name)

"""
This method is based on the part of the gdb's code.
https://github.com/gdb/kaggle/blob/master/denoising-dirty-documents/submit.py
Thanks.
"""
def test_and_save(model):
    x_test, s_test, name_test = import_data()
    y_test = np.zeros((len(name_test), P.max_height, P.max_width))
    f = gzip.open(os.path.join(P.result_dir, P.submission), 'w')
    f.write('id,value\n')
    for i, name in enumerate(name_test):
        printr(name)
        x_batch = x_test[i:i+1, 0:1, 0:s_test[i, 0]+2*P.reduced, 0:s_test[i, 1]+2*P.reduced].astype(np.float32)
        if P.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
        y = forward(x_batch, model)
        if P.gpu >= 0:
            y = cuda.to_cpu(y.data)
        else:
            y = y.data
        assert y.shape[2] == s_test[i, 0] and y.shape[3] == s_test[i, 1]
        y = 1 - y.T
        y = y[:, :, 0, 0]
        y = np.fmax(y, np.zeros(y.shape))
        #y_test[i, 0:s_test[i, 1], 0:s_test[i, 0]] = y
        save_as_image(y, os.path.join(P.result_dir, name))
        it = np.nditer(y, flags=['multi_index'])
        while not it.finished:
            pixel = it[0]
            i, j = it.multi_index
            f.write('{}_{}_{},{}\n'.format(name.replace('.png', ''), i + 1, j + 1, pixel))
            it.iternext()
    f.close()
        # csvファイルとして保存する機能は未実装
def forward(x_data, model):
    x = Variable(x_data)
    h = F.relu(model.conv1(x))
    h = F.relu(model.conv2(h))
    h = F.relu(model.conv3(h))
    h = F.relu(model.conv4(h))
    h = F.relu(model.conv5(h))
    h = F.relu(model.conv6(h))
    return h

"""
def forward(x_data, model):
    x = Variable(x_data)
    h = F.relu(model.conv1(x))
    h = F.relu(model.conv2(h))
    h = F.relu(model.conv3(h))
    h = F.relu(model.conv4(h))
    h = F.relu(model.conv4(h))
    h = F.relu(model.conv5(h))
    return h
"""

def main():
    if P.gpu >= 0:
        cuda.init(P.gpu)
    model = pickle.load(open(os.path.join(P.model_dir, P.model_name), 'rb'))
    test_and_save(model)
    return

if __name__ == '__main__':
    main()
