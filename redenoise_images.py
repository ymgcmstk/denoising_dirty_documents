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
P.result2_dir   = './result_retrain/'
P.model_name    = 'model_retrain_0d152456015581.cPickle'
P.submission    = 'submission.txt.gz'
P.gpu           = 0
P.max_width     = 540
P.max_height    = 420

P.write = False

P.reduced = 4

def import_data():
    test_list = os.listdir(P.result_dir)
    x_test = np.zeros((len(test_list), 1, P.max_width+2*P.reduced, P.max_height+2*P.reduced))
    s_test = np.zeros((len(test_list), 2))
    name_test = []
    count = 0
    for i in test_list:
        if not '.png' in i:
            continue
        input_image = np.array(Image.open(os.path.join(P.result_dir, i)))
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
    if P.write:
        f = gzip.open(os.path.join(P.result2_dir, P.submission), 'w')
        f.write('id,value\n')
    for count, name in enumerate(name_test):
        printr(name)
        x_batch = x_test[count:count+1, 0:1, 0:s_test[count, 0]+2*P.reduced, 0:s_test[count, 1]+2*P.reduced].astype(np.float32)
        if P.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
        y = forward(x_batch, model)
        if P.gpu >= 0:
            y = cuda.to_cpu(y.data)
        else:
            y = y.data
        print y.shape, s_test[count, :]
        assert y.shape[2] == s_test[count, 0] and y.shape[3] == s_test[count, 1]
        y = 1 - y.T
        y = y[:, :, 0, 0]
        y = np.fmax(y, np.zeros(y.shape))
        if P.write:
            it = np.nditer(y, flags=['multi_index'])
            while not it.finished:
                pixel = it[0]
                i, j = it.multi_index
                f.write('{}_{}_{},{}\n'.format(name.replace('.png', ''), i + 1, j + 1, pixel))
                it.iternext()
        input_image = 1 - x_test[count, 0, P.reduced:s_test[count, 0]+P.reduced, P.reduced:s_test[count, 1]+P.reduced].T
        assert y.shape == input_image.shape
        y = np.r_[y, input_image]
        save_as_image(y, os.path.join(P.result2_dir, name))
    if P.write:
        f.close()

def forward(x_data, model):
    x = Variable(x_data)
    h = F.relu(model.conv1(x))
    h = F.relu(model.conv2(h))
    h = F.relu(model.conv3(h))
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