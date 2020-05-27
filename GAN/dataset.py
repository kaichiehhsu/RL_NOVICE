import numpy as np
import os
import pickle
import gzip
import urllib.request

import torch
import torch.nn as nn
from torch.utils.data import Dataset

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def _dense_to_one_hot(labels, num_classes):
    label_oh = np.zeros(shape=(labels.shape[0], num_classes))
    for i, label in enumerate(labels):
        label_oh[i, label] = 1
    return label_oh

def _extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
    Args:
    f: A file object that can be passed into a gzip reader.
    Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].
    Raises:
    ValueError: If the bytestream does not start with 2051.
    """
    #print('Extracting', f)
    with gzip.open(f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                            (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, rows, cols, 1)
        data = (data/255) -0.5
    return data

def _extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index].
    Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.
    Returns:
    labels: a 1D uint8 numpy array.
    Raises:
    ValueError: If the bystream doesn't start with 2049.
    """
    #print('Extracting', f)
    with gzip.open(f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                            (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return _dense_to_one_hot(labels, num_classes)
    return labels

def channel_first(x):
    if x.shape[3] == 1 or x.shape[3] == 3:
        x = np.moveaxis(x, -1, 1)
    return x

class CIFAR_DS:
    def __init__(self, one_hot = True):
        import pickle
        id_list = [1,2,3,4,5]
        train_data = []
        train_labels = []

        for batch_id in id_list:
            with open('data/cifar-10-batches-py/data_batch_' + str(batch_id), mode='rb') as file:
                batch = pickle.load(file, encoding='latin1')
            train_data_batch = np.array(batch['data'])/255 - 0.5
            train_labels_batch = np.array(batch['labels'])
            train_data.extend(train_data_batch)
            train_labels.extend(train_labels_batch)
        train_data = np.array(train_data).reshape((len(train_data),3,32,32))
        train_labels = np.array(train_labels)

        with open('data/cifar-10-batches-py/test_batch', mode='rb') as file:
            batch = pickle.load(file, encoding='latin1')
        self.test_data = np.array(batch['data']).reshape((10000,3,32,32))/255 - 0.5
        test_labels = np.array(batch['labels'])

        if one_hot:
            train_labels = _dense_to_one_hot(train_labels, 10)
            self.test_labels = _dense_to_one_hot(test_labels, 10)
        
        VALIDATION_SIZE = 5000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        if one_hot:
            self.validation_labels = train_labels[:VALIDATION_SIZE, :]
        else:
            self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        if one_hot:
            self.train_labels = train_labels[VALIDATION_SIZE:, :]
        else:
            self.train_labels = train_labels[VALIDATION_SIZE:]
        
        with open('data/cifar-10-batches-py/batches.meta', mode='rb') as file:
            batch = pickle.load(file, encoding='latin1')
        self.label_names = batch['label_names']

class MNIST_DS:
    def __init__(self, one_hot = True):
        if not os.path.exists("data"):
            os.mkdir("data")
            files = ["train-images-idx3-ubyte.gz",
                     "t10k-images-idx3-ubyte.gz",
                     "train-labels-idx1-ubyte.gz",
                     "t10k-labels-idx1-ubyte.gz"]
            for name in files:
                urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)

        train_data   = _extract_images("data/train-images-idx3-ubyte.gz")
        train_labels = _extract_labels("data/train-labels-idx1-ubyte.gz", one_hot)
        self.test_data    = _extract_images("data/t10k-images-idx3-ubyte.gz")
        self.test_labels  = _extract_labels("data/t10k-labels-idx1-ubyte.gz", one_hot)
        
        VALIDATION_SIZE = 5000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        if one_hot:
            self.validation_labels = train_labels[:VALIDATION_SIZE, :]
        else:
            self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        if one_hot:
            self.train_labels = train_labels[VALIDATION_SIZE:, :]
        else:
            self.train_labels = train_labels[VALIDATION_SIZE:]

class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        # from_numpy: share the same memory
        # FloatTensor: copy the data
        self.data = torch.FloatTensor(channel_first(data))
        self.target = torch.FloatTensor(target)
        self.transform = transform
        
    def __getitem__(self, index): # for torch.dataloader
        x = self.data[index]
        y = self.target[index]
        mod = 0
        
        if self.transform:
            x_ = self.transform(x)
            mod = torch.norm(x_ - x, p=2)
            return x_, y
        return x, y
    
    def __len__(self): # for torch.dataloader
        return len(self.data)

    def gen_target(self, num_samples, targeted=True, offset=0):
        imgs = []
        targets = []
        n_classes = self.target.shape[1]
        for i in range(num_samples):
            img =  self.data[offset+i]
            label_oh = self.target[offset+i]
            label = torch.argmax(label_oh)
            if targeted:
                for c in range(n_classes):
                    if c == label:
                        continue
                    else:
                        imgs.append(img)
                        targets.append(torch.eye(n_classes)[c])
            else:
                imgs.append(img)
                targets.append(label_oh)
        imgs = torch.stack(imgs, dim=0)
        targets = torch.stack(targets, dim=0)
        
        return imgs, targets

#==== LeNet Model ====
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 3, 1, 1), #in_c, out_c, ker_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.BatchNorm1d(120),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU())
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1) #flatten to batch_size, -1 
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def init_global(self):
        def init_params(net):
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal(m.weight, mode='fan_out')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant(m.weight, 1)
                    nn.init.constant(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal(m.weight, std=1)
        self.conv1.apply(init_params)
        self.conv2.apply(init_params)
        self.fc1.apply(init_params)
        self.fc2.apply(init_params)
        self.fc3.apply(init_params)
