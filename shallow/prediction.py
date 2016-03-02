import matplotlib.pyplot as plt
import Image
import ImageOps
from scipy import misc
import scipy.io
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet,BatchIterator
import os
import numpy as np
from sklearn.utils import shuffle
import cPickle as pickle
import theano
from scipy import ndimage
import matplotlib.pyplot as plt

MAT_TEST= '/imatge/jpan/work/iSUN/testing.mat'
MAT_VAL = '/imatge/jpan/work/iSUN/validation.mat'

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

f = open('net_sun.pickle', 'rb')
net2=pickle.load(f)
f.close()

def loadNameList(MATFILE,numImg,typee): #numImg:5000 for validation 10.000 for training
  mat = scipy.io.loadmat(MATFILE)
  xd = mat[typee] #training?
  imageList=range(numImg) 
  for i in range(numImg):
    imageList[i] = xd[i,0][0][0]
  return imageList

def loadSaliencyMap(FILENAME):
    saliency = scipy.io.loadmat('/imatge/jpan/work/iSUN/saliency/saliency/'+FILENAME+'.mat')
    mapp =saliency['I']
    return mapp

def newload(FNAME):
    f = file(FNAME, 'rb')
    X,y= pickle.load(f)
    f.close()
    return X

def newload2(FNAME):
    f = file(FNAME, 'rb')
    X= pickle.load(f)
    f.close()
    return X

def predict():
    NumSample_test=2000

    names_test = loadNameList(MAT_TEST,NumSample_test,'testing')
    X_test=newload2('data_iSun_Test.cPickle')
    y_pred_test = net2.predict(X_test)

    print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X_test.shape, X_test.min(), X_test.max()))
    print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y_pred_test.shape, y_pred_test.min(), y_pred_test.max()))
      
    NumSample_val=926

    names_val = loadNameList(MAT_VAL,NumSample_val,'validation')
    X_val=newload('data_iSun_VAL.cPickle')
    y_pred_val = net2.predict(X_val)

    print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X_val.shape, X_val.min(), X_val.max()))
    print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y_pred_val.shape, y_pred_val.min(), y_pred_val.max()))

    for i in range(NumSample_val):
        img = misc.imread('/imatge/jpan/work/iSUN/images/'+names_val[i]+'.jpg')
        tmp = y_pred_val[i].reshape(48,48)
        blured= ndimage.gaussian_filter(tmp, sigma=3)
        y = misc.imresize(blured,(img.shape[0],img.shape[1]))/255.
        d = {'I':y} 
        scipy.io.savemat('/imatge/jpan/work/backup/val_isun/'+names_val[i],d)
  
    for i in range(NumSample_test):
        img = misc.imread('/imatge/jpan/work/iSUN/images/'+names_test[i]+'.jpg')
        #imsave('validation_test/'+names[i]+'.png', y)
        tmp = y_pred_test[i].reshape(48,48)
        blured= ndimage.gaussian_filter(tmp, sigma=3)
        y = misc.imresize(blured,(img.shape[0],img.shape[1]))/255.
        d = {'I':y} 
        scipy.io.savemat('/imatge/jpan/work/backup/result_isun/'+names_test[i],d)





