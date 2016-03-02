#THEANO_FLAGS='floatX=float32,device=gpu,nvcc.fastmath=True' CUDA_ROOT=/usr/local/cuda srun --x11 -w c7 --pty --mem=4000mb --gres=gpu:1  python saliency.py  > ~/logs/jnet.log 2>&1 &
import os
import numpy as np
from sklearn.utils import shuffle
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
import glob

url = './images/' # <------- cambiar la ruta !


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

f = open('./JuntingNet_iSUN.pickle', 'rb')
net2=pickle.load(f)
f.close()
num_img=len(glob.glob(url+"*.jpg"))

def to_rgb(im):
    im.resize((96, 96, 1), refcheck=False)
    return np.repeat(im.astype(np.float32), 3, 2)
    
def loadImage():
    xt = np.zeros((num_img, 3, 96, 96), dtype='float32')
    i = 0
    for file in glob.glob(url+"*.jpg"):   
        img = Image.open(file)  
        img = ImageOps.fit(img, (96, 96), Image.ANTIALIAS)
        img = np.asarray(img, dtype = 'float32') / 255.       
        if(cmp(img.shape , (96,96,3)) == 0):
            img = img.transpose(2,0,1).reshape(3, 96, 96)
            xt[i]=img
        else:
            aux=to_rgb(img)
            aux = aux.transpose(2,0,1).reshape(3, 96, 96)
            xt[i]=aux
        i = i + 1
        return xt
    
def saliencyPredictor(xt):
    y_pred_test = net2.predict(xt)
    i=0
    for file in glob.glob(url+"*.jpg"):   
      img = misc.imread(file)  
      tmp = y_pred_test[i].reshape(48,48)
      blured= ndimage.gaussian_filter(tmp, sigma=3)
      y = misc.imresize(blured,(img.shape[0],img.shape[1]))/255.
      misc.imsave('./saliency/'+os.path.basename(file), y)  
      i= i+1


tmp=loadImage()
print("tmp.shape == {}; tmp.min == {:.3f}; tmp.max == {:.3f}".format(
tmp.shape, tmp.min(), tmp.max()))
saliencyPredictor(tmp)   
     
     