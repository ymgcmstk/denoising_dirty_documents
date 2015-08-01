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
P.model_names   = [
    'model_0d00149540185157_seed_None_ds_128.cPickle',
    'model_0d00159264426111_seed_1_ds_128.cPickle',
    'model_0d0015791014921_seed_2_ds_128.cPickle',
    'model_0d00194685728638_seed_3_ds_128.cPickle',
    'model_0d0016670271234_seed_4_ds_128.cPickle'
]
P.submission    = 'submission_5ens.txt.gz'
P.gpu           = 1
P.use_mean_var  = False
P.max_width     = 540
P.max_height    = 420

P.write = True
P.reduced = 6

def import_data():
    test_list = os.listdir(os.path.join(P.data_dir, 'test'))
    x_test = np.zeros((len(test_list), 1, P.max_width+2*P.reduced, P.max_height+2*P.reduced))
    s_test = np.zeros((len(test_list), 2))
    name_test = []
    count = 0
    for i in test_list:
        if not '.png' in i:
            continue
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
def test_and_save():
    x_test, s_test, name_test = import_data()
    # y_test = np.zeros((len(name_test), P.max_height, P.max_width))
    y_dic  = {}

    for model_count, model_name in enumerate(P.model_names):
        model = pickle.load(open(os.path.join(P.model_dir, model_name), 'rb'))
        for count, name in enumerate(name_test):
            printr('calculating ' + str(model_count) + 'st/nd/th ' + name)
            x_batch = x_test[count:count+1, 0:1, 0:s_test[count, 0]+2*P.reduced, 0:s_test[count, 1]+2*P.reduced].astype(np.float32)
            if P.gpu >= 0:
                x_batch = cuda.to_gpu(x_batch)
            y = forward(x_batch, model)
            if P.gpu >= 0:
                y = cuda.to_cpu(y.data)
            else:
                y = y.data
            assert y.shape[2] == s_test[count, 0] and y.shape[3] == s_test[count, 1]
            y = 1 - y.T
            y = y[:, :, 0, 0]
            y = np.fmax(y, np.zeros(y.shape))
            if not name in y_dic.keys():
                y_dic[name] = np.zeros(y.shape)
            y_dic[name] += y

    if P.write:
        f = gzip.open(os.path.join(P.result_dir, P.submission), 'w')
        f.write('id,value\n')

    for count, name in enumerate(name_test):
        printr('saving ' + name)
        y = y_dic[name] / len(P.model_names)
        if P.write:
            it = np.nditer(y, flags=['multi_index'])
            while not it.finished:
                pixel = it[0]
                i, j = it.multi_index
                f.write('{}_{}_{},{}\n'.format(name.replace('.png', ''), i + 1, j + 1, pixel))
                it.iternext()
        input_image = 1 - x_test[count, 0, P.reduced:s_test[count, 0]+P.reduced, P.reduced:s_test[count, 1]+P.reduced].T
        assert y.shape == input_image.shape
        # y = np.r_[y, input_image]
        save_as_image(y, os.path.join(P.result_dir, name))
    if P.write:
        f.close()

def forward(x_data, model):
    x = Variable(x_data)
    h = F.relu(model.conv1(x))
    h = F.relu(model.conv2(h))
    h = F.relu(model.conv3(h))
    h = F.relu(model.conv4(h))
    h = F.relu(model.conv5(h))
    h = F.relu(model.conv6(h))
    h = F.relu(model.conv7(h))
    h = F.relu(model.conv8(h))
    return h

def main():
    if P.gpu >= 0:
        cuda.init(P.gpu)
    test_and_save()
    return

if __name__ == '__main__':
    main()
