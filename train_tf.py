import mymodel, mymodel_knn,seg_model

import tensorflow as tf
import numpy as np
import time,json
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

def genData(cls,limit=None):
    assert type(cls) is str

    seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                   'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                   'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                   'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15],
                   'Knife': [22, 23]}

    data = np.load("/home/tegs/RGCNN/data_%s.npy" % cls)
    label = np.load("/home/tegs/RGCNN/label_%s.npy" % cls)

    data = data[:limit]
    label = label[:limit]

    seg = {}
    name = {}
    i = 0
    for k,v in sorted(seg_classes.items()):
        for value in v:
            seg[value] = i
            name[value] = k
        i += 1
    cnt = data.shape[0]
    cat = np.zeros((cnt))
    for i in range(cnt):
        cat[i] = seg[label[i][0]]
    return data,label,cat

def train():
    train_data, train_label, train_cat = genData('train')
    val_data, val_label, val_cat = genData('val')
    test_data, test_label, test_cat = genData('test')
    params = dict()
    params['dir_name'] = 'model'
    params['num_epochs'] = 50
    params['batch_size'] = 26
    params['eval_frequency'] = 30

    # Building blocks.
    params['filter'] = 'chebyshev5'
    params['brelu'] = 'b1relu'
    params['pool'] = 'apool1'

    # Number of classes.
    # C = y.max() + 1
    # assert C == np.unique(y) .size

    # Architecture.
    params['F'] = [128, 512, 1024, 512, 128, 50]  # Number of graph convolutional filters.
    params['K'] = [6, 5, 3, 1, 1, 1]  # Polynomial orders.
    params['M'] = [384, 16, 1]  # Output dimensionality of fully connected layers.

    # Optimization.
    params['regularization'] = 1e-9
    params['dropout'] = 1
    params['learning_rate'] = 1e-3
    params['decay_rate'] = 0.95
    params['momentum'] = 0
    params['decay_steps'] = train_data.shape[0] / params['batch_size']

    model = seg_model.rgcnn(2048, **params)
    accuracy, loss, t_step = model.fit(train_data, train_cat, train_label, val_data, val_cat, val_label,
                                       is_continue=False)

def test():
    test_data, test_label, test_cat = genData('test')
    params = dict()
    params['dir_name'] = 'model'
    params['num_epochs'] = 50
    params['batch_size'] = 26
    params['eval_frequency'] = 30

    # Building blocks.
    params['filter'] = 'chebyshev5'
    params['brelu'] = 'b1relu'
    params['pool'] = 'apool1'

    # Number of classes.
    # C = y.max() + 1
    # assert C == np.unique(y) .size

    # Architecture.
    params['F'] = [128, 512, 1024, 512, 128, 50]  # Number of graph convolutional filters.
    params['K'] = [6, 5, 3, 1, 1, 1]  # Polynomial orders.
    params['M'] = [384, 16, 1]  # Output dimensionality of fully connected layers. For classification only

    # Optimization.
    params['regularization'] = 1e-9
    params['dropout'] = 1
    params['learning_rate'] = 1e-3
    params['decay_rate'] = 0.95
    params['momentum'] = 0
    params['decay_steps'] = test_data.shape[0] / params['batch_size']

    model = seg_model.rgcnn(2048, **params)
    model.evaluate(test_data,test_cat,test_label)


if __name__=="__main__":
    train()