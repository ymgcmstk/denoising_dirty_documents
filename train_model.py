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
import math
import random

P               = edict({})
P.data_dir      = './data/'
P.cache_dir     = './cache/'
P.model_dir     = './models/'
P.model_name    = None
P.prefix        = ''
P.gpu           = 1
P.num_val       = 16 # 16
P.datasize      = 128 # 144
P.random_seed   = 1
P.use_mean_var  = False
P.test_interval = 10
P.disp_num      = 50
P.max_width     = 540
P.max_height    = 420

P.reduced       = 6
P.import_again  = True

P.max_iter      = pow(10, 5)
P.batchsize     = 32
P.edge          = 48
P.lr            = 0.01
P.momentum      = 0.9
P.decay         = 0.0005
P.drop          = 0
P.step_size     = 0.1
P.epoch_count   = 1500

P.add_noise     = 0

def augment_data(x, m, y, s, deterministic=False):
    if deterministic:
        num_for_seed = int(np.random.random()*pow(2,16))
        np.random.seed(0)
    x_batch = np.zeros((P.batchsize, 1, P.edge, P.edge)).astype(np.float32)
    m_batch = np.zeros((P.batchsize, 2, P.edge-2*P.reduced, P.edge-2*P.reduced)).astype(np.float32)
    y_batch = np.zeros((P.batchsize, 1, P.edge-2*P.reduced, P.edge-2*P.reduced)).astype(np.float32)
    for j in range(s.shape[0]):
        start_x = np.random.randint(s[j, 0] - P.edge + 2 * P.reduced)
        start_y = np.random.randint(s[j, 1] - P.edge + 2 * P.reduced)
        x_batch[j, 0, :] = x[j:j+1, 0:1, start_x:start_x+P.edge, start_y:start_y+P.edge]
        m_batch[j, 0:2, :] = m[j:j+1, 0:2, start_x+P.reduced:start_x+P.edge-P.reduced, start_y+P.reduced:start_y+P.edge-P.reduced]
        y_batch[j, 0, :] = y[j:j+1, 0:1, start_x+P.reduced:start_x+P.edge-P.reduced, start_y+P.reduced:start_y+P.edge-P.reduced]
    if P.add_noise > 0:
        x_batch += P.add_noise * np.random.random(x_batch.shape)
        x_batch = np.fmax(x_batch, np.zeros(x_batch.shape))
        x_batch = np.fmin(x_batch, np.ones(x_batch.shape)).astype(np.float32)
    if deterministic:
        np.random.seed(num_for_seed)
    return x_batch, m_batch, y_batch

def import_data(again=False, random_seed=None):
    start_time = time()
    if os.path.exists(os.path.join(P.cache_dir, 'x_train.npy')) and not again:
        x_train = np.load(os.path.join(P.cache_dir, 'x_train.npy'))
        m_train = np.load(os.path.join(P.cache_dir, 'm_train.npy'))
        y_train = np.load(os.path.join(P.cache_dir, 'y_train.npy'))
        s_train = np.load(os.path.join(P.cache_dir, 's_train.npy'))
        x_val   = np.load(os.path.join(P.cache_dir, 'x_val.npy'))
        m_val   = np.load(os.path.join(P.cache_dir, 'm_val.npy'))
        y_val   = np.load(os.path.join(P.cache_dir, 'y_val.npy'))
        s_val   = np.load(os.path.join(P.cache_dir, 's_val.npy'))
        print '[import_data] elapsed time :', elapsed
        return x_train, m_train, y_train, s_train, x_val, m_val, y_val, s_val
    train_list = os.listdir(os.path.join(P.data_dir, 'train'))
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(train_list)
    x_train = np.zeros((P.datasize, 1, P.max_width+2*P.reduced, P.max_height+2*P.reduced))
    m_train = np.zeros((P.datasize, 2, P.max_width+2*P.reduced, P.max_height+2*P.reduced)).astype(np.float32)
    y_train = np.zeros((P.datasize, 1, P.max_width+2*P.reduced, P.max_height+2*P.reduced))
    s_train = np.zeros((P.datasize, 2))

    for count, i in enumerate(train_list[:P.datasize]):
        input_image  = np.array(Image.open(os.path.join(P.data_dir, 'train', i)))
        input_image = 1 - input_image.astype(np.float32).T / 255
        output_image = np.array(Image.open(os.path.join(P.data_dir, 'train_cleaned', i)))
        output_image = 1 - output_image.astype(np.float32).T / 255
        assert input_image.shape == output_image.shape
        s_train[count, 0] = input_image.shape[0]
        s_train[count, 1] = input_image.shape[1]
        x_train[count:count+1, 0:1, P.reduced:s_train[count, 0]+P.reduced, P.reduced:s_train[count, 1]+P.reduced] = input_image
        m_train[count:count+1, 0:1, P.reduced:s_train[count, 0]+P.reduced, P.reduced:s_train[count, 1]+P.reduced] = np.repeat(input_image.mean(axis=0)[:, np.newaxis], s_train[count, 0], axis=1).T
        m_train[count:count+1, 1:2, P.reduced:s_train[count, 0]+P.reduced, P.reduced:s_train[count, 1]+P.reduced] = np.repeat(input_image.var(axis=0)[:, np.newaxis], s_train[count, 0], axis=1).T
        y_train[count:count+1, 0:1, P.reduced:s_train[count, 0]+P.reduced, P.reduced:s_train[count, 1]+P.reduced] = output_image
    num_val = len(train_list) - P.datasize

    x_val = np.zeros((num_val, 1, P.max_width+2*P.reduced, P.max_height+2*P.reduced))
    m_val = np.zeros((num_val, 2, P.max_width+2*P.reduced, P.max_height+2*P.reduced))
    y_val = np.zeros((num_val, 1, P.max_width+2*P.reduced, P.max_height+2*P.reduced))
    s_val = np.zeros((num_val, 2))

    for count, i in enumerate(train_list[P.datasize:]):
        input_image  = np.array(Image.open(os.path.join(P.data_dir, 'train', i)))
        input_image = 1 - input_image.astype(np.float32).T / 255
        output_image = np.array(Image.open(os.path.join(P.data_dir, 'train_cleaned', i)))
        output_image = 1 - output_image.astype(np.float32).T / 255
        assert input_image.shape == output_image.shape
        s_val[count, 0] = input_image.shape[0]
        s_val[count, 1] = input_image.shape[1]
        x_val[count:count+1, 0:1, P.reduced:s_val[count, 0]+P.reduced, P.reduced:s_val[count, 1]+P.reduced] = input_image
        m_val[count:count+1, 0:1, P.reduced:s_val[count, 0]+P.reduced, P.reduced:s_val[count, 1]+P.reduced] = np.repeat(input_image.mean(axis=0)[:, np.newaxis], s_val[count, 0], axis=1).T
        m_val[count:count+1, 1:2, P.reduced:s_val[count, 0]+P.reduced, P.reduced:s_val[count, 1]+P.reduced] = np.repeat(input_image.var(axis=0)[:, np.newaxis], s_val[count, 0], axis=1).T
        y_val[count:count+1, 0:1, P.reduced:s_val[count, 0]+P.reduced, P.reduced:s_val[count, 1]+P.reduced] = output_image
    np.save(os.path.join(P.cache_dir, 'x_train'), x_train)
    np.save(os.path.join(P.cache_dir, 'm_train'), m_train)
    np.save(os.path.join(P.cache_dir, 'y_train'), y_train)
    np.save(os.path.join(P.cache_dir, 's_train'), s_train)
    np.save(os.path.join(P.cache_dir, 'x_val'), x_val.astype(np.float32))
    np.save(os.path.join(P.cache_dir, 'm_val'), m_val.astype(np.float32))
    np.save(os.path.join(P.cache_dir, 'y_val'), y_val.astype(np.float32))
    np.save(os.path.join(P.cache_dir, 's_val'), s_val)
    elapsed = time() - start_time
    print '[import_data] elapsed time :', elapsed
    return x_train, m_train, y_train, s_train, x_val.astype(np.float32), m_val.astype(np.float32), y_val.astype(np.float32), s_val

def train(model, optimizer):
    x_train, m_train, y_train, s_train, x_val, m_val, y_val, s_val = import_data(P.import_again, P.random_seed)
    iter_num     = 0
    min_loss_val = float("inf")

    drop_count = 0
    sum_loss   = 0
    for epoch_num in range(P.epoch_count):
        perm = np.random.permutation(P.datasize)
        for i in range(P.datasize/P.batchsize):
            iter_num += 1
            x_batch = x_train[perm[P.batchsize*i:P.batchsize*(i+1)]]
            m_batch = m_train[perm[P.batchsize*i:P.batchsize*(i+1)]]
            y_batch = y_train[perm[P.batchsize*i:P.batchsize*(i+1)]]
            s_batch = s_train[perm[P.batchsize*i:P.batchsize*(i+1)]]
            x_batch, m_batch, y_batch = augment_data(x_batch, m_batch, y_batch, s_batch)
            if P.gpu >= 0:
                x_batch = cuda.to_gpu(x_batch)
                m_batch = cuda.to_gpu(m_batch)
                y_batch = cuda.to_gpu(y_batch)
            optimizer.zero_grads()
            loss = forward(x_batch, m_batch, y_batch, model)
            optimizer.weight_decay(P.decay)
            loss.backward()
            optimizer.update()
            sum_loss += float(cuda.to_cpu(loss.data)) * P.batchsize
        print 'epoch : ' + str(epoch_num + 1) + ', loss : ' + str(sum_loss)
        sum_loss = 0

        if (epoch_num + 1) == 1000:
            optimizer.lr = 0.001

        if (epoch_num + 1) % P.test_interval is not 0:
            continue

        if P.datasize >= 144:
            if (epoch_num + 1) % 200 == 0:
                fname = 'model_full_' + str(epoch_num + 1) + P.prefix + '.cPickle'
                pickle.dump(model, open(os.path.join(P.model_dir, fname), 'wb'), -1)
                print 'This model has been saved as', fname, '(epoch :', epoch_num + 1, ')'
            continue

        sum_loss_val = 0
        for i in range(x_val.shape[0]):
            x_batch_temp = x_val[i:i+1, 0:1, 0:s_val[i, 0]/2+2*P.reduced, 0:s_val[i, 1]+2*P.reduced]
            x_batch = np.zeros(x_batch_temp.shape).astype(np.float32)
            x_batch[:] = x_batch_temp[:]
            m_batch_temp = m_val[i:i+1, 0:2, P.reduced:s_val[i, 0]/2+P.reduced, P.reduced:s_val[i, 1]+P.reduced]
            m_batch = np.zeros(m_batch_temp.shape).astype(np.float32)
            m_batch[:] = m_batch_temp[:]
            y_batch_temp = y_val[i:i+1, 0:1, P.reduced:s_val[i, 0]/2+P.reduced, P.reduced:s_val[i, 1]+P.reduced]
            y_batch = np.zeros(y_batch_temp.shape).astype(np.float32)
            y_batch[:] = y_batch_temp[:]
            if P.gpu >= 0:
                x_batch = cuda.to_gpu(x_batch)
                m_batch = cuda.to_gpu(m_batch)
                y_batch = cuda.to_gpu(y_batch)
            optimizer.zero_grads()
            loss = forward(x_batch, m_batch, y_batch, model)
            sum_loss_val += float(cuda.to_cpu(loss.data))

            x_batch_temp = x_val[i:i+1, 0:1, s_val[i, 0]/2:s_val[i, 0]+2*P.reduced, 0:s_val[i, 1]+2*P.reduced]
            x_batch = np.zeros(x_batch_temp.shape).astype(np.float32)
            x_batch[:] = x_batch_temp[:]
            m_batch_temp = m_val[i:i+1, 0:2, s_val[i, 0]/2+P.reduced:s_val[i, 0]+P.reduced, P.reduced:s_val[i, 1]+P.reduced]
            m_batch = np.zeros(m_batch_temp.shape).astype(np.float32)
            m_batch[:] = m_batch_temp[:]
            y_batch_temp = y_val[i:i+1, 0:1, s_val[i, 0]/2+P.reduced:s_val[i, 0]+P.reduced, P.reduced:s_val[i, 1]+P.reduced]
            y_batch = np.zeros(y_batch_temp.shape).astype(np.float32)
            y_batch[:] = y_batch_temp[:]
            if P.gpu >= 0:
                x_batch = cuda.to_gpu(x_batch)
                m_batch = cuda.to_gpu(m_batch)
                y_batch = cuda.to_gpu(y_batch)
            optimizer.zero_grads()
            loss = forward(x_batch, m_batch, y_batch, model)
            sum_loss_val += float(cuda.to_cpu(loss.data))

        sum_loss_val /= x_val.shape[0]
        if sum_loss_val < min_loss_val:
            min_loss_val = sum_loss_val
            fname = 'model_' + str(min_loss_val).replace('.', 'd') + '_seed_' + str(P.random_seed) + '_ds_' + str(P.datasize) + P.prefix + '.cPickle'
            if drop_count > 0:
                pickle.dump(model, open(os.path.join(P.model_dir, fname), 'wb'), -1)
                print 'This model has been saved as', fname, '(epoch :', epoch_num + 1, ')'
                drop_count = 0
            else:
                print 'This model should have been saved as', fname, '(epoch :', epoch_num + 1, ')'
        else:
            print 'sum_loss_val:', sum_loss_val
            drop_count += 1
            if drop_count == P.drop:
                optimizer.lr *= 0.1
                drop_count = 0

def forward(x_data, m_data, y_data, model):
    x = Variable(x_data)
    if P.use_mean_var:
        m = Variable(m_data)
    y = Variable(y_data)
    h = F.relu(model.conv1(x))
    h = F.relu(model.conv2(h))
    h = F.relu(model.conv3(h))
    h = F.relu(model.conv4(h))
    h = F.relu(model.conv5(h))
    h = F.relu(model.conv6(h))
    if P.use_mean_var:
        h = F.relu(model.conv7(F.concat([m, h])))
    else:
        h = F.relu(model.conv7(h))
    h = F.relu(model.conv8(h))
    return F.mean_squared_error(h, y)

def forward2(x_data, y_data, model):
    x = Variable(x_data)
    y = Variable(y_data)
    h = F.relu(model.conv1(x))
    h = F.relu(model.conv2(h))
    h = F.relu(model.conv3(h))
    h = F.relu(model.conv4(h))
    h = F.relu(model.conv5(h))
    h = F.relu(model.conv6(h))
    y_size = y_data.shape[-2] * y_data.shape[-1]
    assert y_data.shape[-1] > 10
    assert y_data.shape[-2] > 10
    h_np = cuda.to_cpu(h.data)
    h_min = np.fmin(h_np, np.ones(h_np.shape)) - y_data
    return np.linalg.norm(h_min)/math.sqrt(y_size)

def main():
    if P.use_mean_var:
        conv6_output = 126
    else:
        conv6_output = 128

    if P.model_name is None:
        model = FunctionSet(
            conv1 = F.Convolution2D( 1, 128, 3, stride=1),
            conv2 = F.Convolution2D(128, 128, 3, stride=1),
            conv3 = F.Convolution2D(128, 128, 3, stride=1),
            conv4 = F.Convolution2D(128, 128, 3, stride=1),
            conv5 = F.Convolution2D(128, 128, 3, stride=1),
            conv6 = F.Convolution2D(128, conv6_output, 3, stride=1),
            conv7 = F.Convolution2D(128, 128, 1, stride=1),
            conv8 = F.Convolution2D(128, 1, 1, stride=1)
            )
        if P.gpu >= 0:
            cuda.init(P.gpu)
            model.to_gpu()
    else:
        if P.gpu >= 0:
            cuda.init(P.gpu)
        model = pickle.load(open(os.path.join(P.model_dir, P.model_name), 'rb'))

    optimizer = optimizers.MomentumSGD(lr=P.lr, momentum=P.momentum)
    optimizer.setup(model.collect_parameters())

    train(model, optimizer)
    return

if __name__ == '__main__':
    main()
