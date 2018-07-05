import torch.tensor
import torch.nn as nn
import torch.nn.functional as F
import os, time, collections, shutil, numpy as np
from model import *
from collections import defaultdict

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

    # train_data = np.load("/home/tegs/RGCNN/data_train.npy")
    # train_label = np.load("/home/tegs/RGCNN/label_train.npy")
    # val_data = np.load("/home/tegs/RGCNN/data_val.npy")
    # val_label = np.load("/home/tegs/RGCNN/label_val.npy")
    # test_data = np.load("/home/tegs/RGCNN/data_test.npy")
    # test_label = np.load("/home/tegs/RGCNN/label_test.npy")

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

def runEpoch(data,label,cat,model,dev,lr,is_training=True):
    batch_size = model.batch_size
    steps = int(data.shape[0] / batch_size)
    indices = collections.deque()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loss = 0
    res = []
    with torch.cuda.device(dev):
        for i in range(steps):
            if len(indices) < batch_size:
                indices.extend(np.random.permutation(data.shape[0]))
            idx = [indices.popleft() for _ in range(batch_size)]
            batch_data, batch_cat, batch_labels = torch.tensor(data[idx, :], dtype=torch.float).cuda(), torch.tensor(
                cat[idx],dtype=torch.long).cuda(), torch.tensor(label[idx],dtype=torch.long).cuda()
            #     train_batch = torch.tensor(train_data[1:3]).cuda()
            #     cat_batch = torch.tensor(cat_train[1:3],dtype=torch.long).cuda()
            #     label_batch = torch.tensor(train_label[1:3],dtype=torch.long).cuda()
            outputs = model(batch_data, batch_cat)#.permute(0, 2, 1)
            print(outputs)
            print(outputs.size())
            optimizer.zero_grad()
            loss = criterion(outputs, batch_labels)
            if is_training:
                loss.backward()
                optimizer.step()
                print(loss.item())
    return loss

def predict(data,cat,label,model,dev):
    batch_size = model.batch_size
    size = data.shape[0]
    res = np.zeros_like(label)
    with torch.cuda.device(dev):
        for i in range(0,size,model.batch_size):
            batch_data, batch_cat, batch_label = torch.zeros(batch_size,data.shape[1],data.shape[2]).cuda(),torch.zeros(batch_size,cat.shape[1],dtype=torch.long).cuda(),torch.zeros(batch_size,cat.shape[1],dtype=torch.long).cuda()
            batch_data[min(i,i+batch_size)] = data[i:min(i+batch_size,size)]
            batch_cat[min(i,i+batch_size)] = cat[i:min(i+batch_size,size)]
            batch_label[min(i,i+batch_size)] = label[i:min(i+batch_size,size)]
            # batch_data, batch_cat, batch_labels = torch.tensor(data[i:i+batch_size,:]).cuda(), torch.tensor(
            #     cat[i:min(i+batch_size,], dtype=torch.long).cuda(), torch.tensor(label[i:min(i+batch_size,size)], dtype=torch.long).cuda()
            outputs = model(batch_data, batch_cat)
            res[i:min(i+batch_size,size)] = outputs.numpy()
    return res

def evaluate(data,cat,labels,model,dev):
    seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                        'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                        'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                        'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                        'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
    label_to_cat = {}
    for key in seg_classes.keys():
        for label in seg_classes[key]:
            label_to_cat[label] = key

    predictions = predict(data,cat,labels,model,dev)
    ncorrects = np.mean(predictions == labels, axis=1)
    print(ncorrects)
    accuracy = np.mean(ncorrects)
    f1 = 0

    cat_iou = defaultdict(list)

    tot_iou = []
    for i in range(predictions.shape[0]):
        segp = predictions[i, :]
        segl = labels[i, :]
        cat = label_to_cat[segl[0]]
        part_ious = [0.0 for _ in range(len(seg_classes[cat]))]

        for l in seg_classes[cat]:
            if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                part_ious[l - seg_classes[cat][0]] = 1.0
            else:
                part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                    np.sum((segl == l) | (segp == l)))
        cat_iou[cat].append(np.mean(part_ious))
        tot_iou.append(np.mean(part_ious))

    for key, value in cat_iou.items():
        print(key + ': {:.4f}, total: {:d}'.format(np.mean(value), len(value)))
    # print(tot_iou)
    # accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
    # f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average='weighted')
    string = 'accuracy: {:.4f} ({:d} / {:d}), iou (weighted): {:.4f}'.format(
        accuracy, np.sum(predictions == labels), labels.shape[0] * labels.shape[1], np.mean(tot_iou))
    return string, accuracy, f1

def train():
    train_data,train_label,train_cat = genData('train',limit=300)
    val_data,val_label,val_cat = genData('val',limit=300)

    # train_data, train_label, train_cat = train_data[:400], train_label[:400], train_cat[:400]
    dev = 0
    params = dict()
    params['vertice'] = 2048
    params['F'] = [260, 180, 50]  # Number of graph convolutional filters.
    params['K'] = [6, 5, 3]  # Polynomial orders.
    params['M'] = [128, 50]
    params['batch_size'] = 30
    num_epochs = 20

    model = RGCNN_Cls(**params).cuda(dev)
    for epoch in range(num_epochs):
        loss = runEpoch(train_data,train_label,train_cat,model,dev,1e-4)

        print("Train loss of Epoch %d: %f" % (epoch,loss))
        loss = evaluate(val_data,val_label,val_cat,model,3)
        print("Valid loss: %f" % loss)

if __name__=='__main__':
    train()