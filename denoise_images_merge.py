# -*- coding:utf-8 -*-
from mytoolbox import *
import cPickle as pickle
import numpy as np
import Image
import os
from easydict import EasyDict as edict
from time import time
import gzip

P               = edict({})
P.data_dir      = './data/'
P.cache_dir     = './cache/'
P.model_dir     = './models/'
P.result_dirs   = ['./result/', './result_2/']
P.result_dir    = './result_0/'
P.model_name    = 'model_0d0016479307742.cPickle'
P.submission    = 'submission.txt.gz'
P.gpu           = 0
P.max_width     = 540
P.max_height    = 420

P.again = False
P.write = False
P.reduced = 6

def import_data():
    test_list = os.listdir(os.path.join(P.data_dir, 'test'))
    x_test = np.zeros((len(test_list), 1, P.max_width+2*P.reduced, P.max_height+2*P.reduced))
    s_test = np.zeros((len(test_list), 2))
    name_test = []
    count = 0
    f = gzip.open(os.path.join(P.result_dir, P.submission), 'w')
    f.write('id,value\n')
    for count, fname in enumerate(test_list):
        printr(str(count) + '/' + str(len(test_list)))
        if not '.png' in fname:
            continue
        input_image = np.array(Image.open(os.path.join(P.result_dirs[0], fname))).astype(np.float32)
        for dir_name in P.result_dirs[1:]:
            input_image += np.array(Image.open(os.path.join(dir_name, fname)))
        input_image /= len(P.result_dirs)
        save_as_image(input_image.astype(np.uint8), os.path.join(P.result_dir, fname))
        it = np.nditer(input_image/255, flags=['multi_index'])
        temp_str = ''
        while not it.finished:
            pixel = it[0]
            i, j = it.multi_index
            f.write('{}_{}_{},{}\n'.format(fname.replace('.png', ''), i + 1, j + 1, pixel))
            it.iternext()
    f.close()

def save_as_image(mat, name):
    assert len(mat.shape) == 2
    img = Image.fromarray(mat)
    img.save(name)

"""
This method is based on the part of the gdb's code.
https://github.com/gdb/kaggle/blob/master/denoising-dirty-documents/submit.py
Thanks.
"""

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
    import_data()
    return

if __name__ == '__main__':
    main()
