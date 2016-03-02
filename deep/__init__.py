import os
import sys
import cv2
import numpy as np

#PYCAFFE_DIR = '/home/kevin/Development/caffe3/python'
PYCAFFE_DIR = '/usr/local/opt/caffe-2015-07/python'


def _create_net(specfile, modelfile):
    if not PYCAFFE_DIR in sys.path:
        sys.path.insert(0, PYCAFFE_DIR)
    import caffe
    return caffe.Net(specfile, modelfile, caffe.TEST)
    
def find_scale_to_fit(im, shape):
    """Finds the scale that makes the image fit in the rect"""
    w, h = im.shape[1], im.shape[0]
    target_w, target_h = shape[1], shape[0]
    scale = 1.0
    if target_w is not None:
        scale = min(scale, target_w / float(w))
    if target_h is not None:
        scale = min(scale, target_h / float(h))
    return scale

class SalNet(object):
    input_layer = 'data1'
    output_layer = 'deconv1'
    default_model_path = os.path.join(os.path.dirname(__file__), 'model')
    
    def __init__(self, 
        specfile=None, 
        modelfile=None, 
        input_size=None,
        max_input_size=(320, 320),
        channel_swap=(2,1,0),
        mean_value=(100,110,118),
        input_scale=0.0078431372549,
        saliency_mean=127,
        blur_size=5,
        stretch_output=True,
        interpolation=cv2.INTER_CUBIC):
        
        if not specfile:
            specfile = os.path.join(self.default_model_path, 'deploy.prototxt')
            
        if not modelfile:
            modelfile = os.path.join(self.default_model_path, 'model.caffemodel')
        
        self.input_size = input_size
        self.max_input_size = max_input_size
        self.channel_swap = channel_swap
        self.mean_value = mean_value
        self.input_scale = input_scale
        self.saliency_mean = saliency_mean
        self.blur_size = blur_size
        self.net = _create_net(specfile, modelfile)
        self.stretch_output = stretch_output
        self.interpolation = interpolation
        
    def scale_input(self, im):
        if self.input_size:
            # scale to fixed size
            h, w = self.input_size
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        elif self.max_input_size:
            # scale to fit in a rectangle
            scale = find_scale_to_fit(im, self.max_input_size)
            im = cv2.resize(im, None, fx=scale, fy=scale)
        return im
        
    def preprocess_input(self, input_image):
        # scale
        im = self.scale_input(input_image)
        # rgb -> bgr
        if self.channel_swap:
            im = im[:,:,self.channel_swap]
        # to float
        im = im.astype(np.float32)
        # mean subtraction
        im -= np.array(self.mean_value)
        # scale to [-1,1]
        im *= self.input_scale
        # transpose
        im = im.transpose((2,0,1))
        # add lead dimension
        return np.ascontiguousarray(im[np.newaxis,...], dtype=np.float32)
    
    def postprocess_output(self, net_output, map_shape):
        # squeeze extra dimensions
        p = np.squeeze(np.array(net_output))
        # rescale
        p *= 128
        # add back the mean
        p += self.saliency_mean 
        # clip
        p = np.clip(p, 0, 255)
        # resize back to original size
        if map_shape:
            h, w = map_shape
            p = cv2.resize(p, (w, h), interpolation=self.interpolation)   
        # blur
        if self.blur_size:
            p = cv2.GaussianBlur(p, (self.blur_size, self.blur_size), 0)
        # clip again
        p = np.clip(p, 0, 255)
        # stretch
        if self.stretch_output:
            if p.max() > 0:
                p = (p / p.max()) * 255.0 
        return p.astype(np.uint8)
    
    def get_saliency(self, image):
        
        # Prepare the image for the network
        net_input = self.preprocess_input(image)
        
        # Reshape the input layer to match the network input
        self.net.blobs[self.input_layer].reshape(*net_input.shape)
        
        # Copy the prepared image to the network input layer
        self.net.blobs[self.input_layer].data[...] = net_input
        
        # Run the network forward
        self.net.forward()
        
        # Grab the output layer
        net_output = self.net.blobs[self.output_layer].data[0,0]
        
        # Postprocess the output to compute saliency map
        return self.postprocess_output(net_output, image.shape[:2])
        
    