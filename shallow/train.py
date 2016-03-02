# add to kfkd.py
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet,BatchIterator
import os
import numpy as np
from sklearn.utils import shuffle
import cPickle as pickle
import matplotlib.pyplot as plt
import Image
import ImageOps
from scipy import misc
import scipy.io
import theano

def load():
    f = file('data_Salicon_T.cPickle', 'rb')
    loaded_obj = pickle.load(f)
    f.close()
    X, y = loaded_obj
    return X, y

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]
        tmp =  yb[indices].reshape(bs/2,1,48,48)
        mirror = tmp[ :,:,:, ::-1]
        yb[indices] =  mirror.reshape(bs/2,48*48)
        return Xb, yb


net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('maxout6',layers.FeaturePoolLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(5, 5), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(3, 3), pool2_pool_size=(2, 2),
    conv3_num_filters=64, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2),
    hidden4_num_units=48*48*2, 
    maxout6_pool_size=2,output_num_units=48*48,output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.05)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,

    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.05, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    max_epochs=1200,
    verbose=1,
    )

X, y = load() 

print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))

X = X.astype(np.float32)
y = y.astype(np.float32)
net.fit(X, y)

with open('JuntingNet_SALICON.pickle', 'wb') as f:
   pickle.dump(net2, f, -1)