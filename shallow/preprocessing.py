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


MAT_TRAIN_SUN = '/imatge/jpan/work/iSUN/training.mat'
MAT_VAL_SUN = '/imatge/jpan/work/iSUN/validation.mat'    
MAT_TEST_SUN = '/imatge/jpan/work/iSUN/testing.mat'    
    
def loadNameListSUN(MATFILE,numImg,typee):
    mat = scipy.io.loadmat(MATFILE)
    xd = mat[typee] #training?
    imageList=range(numImg) 
    for i in range(numImg):
      imageList[i] = xd[i,0][0][0]
    return imageList 

def loadSaliencyMapSUN(FILENAME):
    saliency = scipy.io.loadmat('/imatge/jpan/work/iSUN/saliency/saliency/'+FILENAME+'.mat')
    mapp =saliency['I']
    return mapp

def to_rgb(im):
    im.resize((96, 96, 1), refcheck=False)
    return np.repeat(im.astype(np.float32), 3, 2)
    
def loadSUN():
    NumSample = 6000;
    X1 = np.zeros((NumSample, 3, 96, 96), dtype='float32')
    y1 = np.zeros((NumSample,48*48), dtype='float32')

    names = loadNameListSUN(MAT_TRAIN_SUN,NumSample,'training')
    for i in range(NumSample):
        img = Image.open('/imatge/jpan/work/iSUN/images/'+names[i]+'.jpg')
        img = ImageOps.fit(img, (96, 96), Image.ANTIALIAS)
        img = np.asarray(img, dtype = 'float32') /255.

        if(cmp(img.shape , (96,96,3)) == 0):
            img = img.transpose(2,0,1).reshape(3, 96, 96)
            X1[i] = img
        else:
            print names[i]
            aux=to_rgb(img)
            aux = aux.transpose(2,0,1).reshape(3, 96, 96)
            X1[i]=aux
            
        label = loadSaliencyMapSUN(names[i])
        label = misc.imresize(label,(48,48)) / 127.5
        label = label -1.
        y1[i] =  label.reshape(1,48*48)  
  
    data_to_save = (X1, y1)
    f = file('data_iSun_T.cPickle', 'wb')
    pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    

def loadiSUNTest():    
    NumSample = 2000
    names = loadNameListSUN(MAT_TEST_SUN,NumSample,'testing')
    Xt = np.zeros((NumSample, 3, 96, 96), dtype='float32')
    for i in range(NumSample):
        img = Image.open('/imatge/jpan/work/iSUN/images/'+names[i]+'.jpg')
        img = ImageOps.fit(img, (96, 96), Image.ANTIALIAS)
        img = np.asarray(img, dtype = 'float32') / 255.
        
        if(cmp(img.shape , (96,96,3)) == 0):
            img = img.transpose(2,0,1).reshape(3, 96, 96)
            Xt[i] = img
        else:
            print names[i]
            aux=to_rgb(img)
            aux = aux.transpose(2,0,1).reshape(3, 96, 96)
            Xt[i]=aux
            
    data_to_save = Xt
    f = file('data_iSUN_Test.cPickle', 'wb')
    pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def loadiSUNVAL():
    NumSample = 926;
    X2 = np.zeros((NumSample, 3, 96, 96), dtype='float32')
    y2 = np.zeros((NumSample,48*48), dtype='float32')
    names = loadNameListSUN(MAT_VAL_SUN,NumSample,'validation')
    for i in range(NumSample):
        img = Image.open('/imatge/jpan/work/iSUN/images/'+names[i]+'.jpg')
        img = ImageOps.fit(img, (96, 96), Image.ANTIALIAS)
        img = np.asarray(img, dtype = 'float32') /255.

        if(cmp(img.shape , (96,96,3)) == 0):
            img = img.transpose(2,0,1).reshape(3, 96, 96)
            X2[i] = img
        else:
            print names[i]
            aux=to_rgb(img)
            aux = aux.transpose(2,0,1).reshape(3, 96, 96)
            X2[i]=aux
            
        label = loadSaliencyMapSUN(names[i])
        label = misc.imresize(label,(48,48)) / 127.5
        label = label -1.
        y2[i] =  label.reshape(1,48*48)    
  
    data_to_save = (X2, y2)
    f = file('data_iSun_VAL.cPickle', 'wb')
    pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def load():
    f = file('data_iSUN_T.cPickle', 'rb')
    loaded_obj = pickle.load(f)
    f.close()
    X, y = loaded_obj
    return X, y

def loadmirror():      
	X, y = load() 

	print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
	    X.shape, X.min(), X.max()))
	print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
	    y.shape, y.min(), y.max()))

	X = X.astype(np.float32)
	y = y.astype(np.float32)

	Xm = X[:, :, :, ::-1]

	tmp =  y.reshape(9999,1,48,48)
	mirror = tmp[ :,:, :,::-1]
	ym =  mirror.reshape(9999,48*48)

	print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
	    Xm.shape, Xm.min(), Xm.max()))
	print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
	    ym.shape, ym.min(), ym.max()))

	data_to_save = (Xm, ym)
	f = file('data_Salicon_mirror.cPickle', 'wb')
	pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
	f.close()
 