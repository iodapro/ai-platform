from __future__ import division
import tensorflow as tf
from tensorflow.python.keras import models 
from abc import ABCMeta, abstractmethod

# interface for neural style transfer model 
class ModelInterfaceForNST:
    __metaclass__ = ABCMeta
    @abstractmethod
    def preprocess_input(self, img):
        raise NotImplementedError
    @abstractmethod
    def get_model(self):
        raise NotImplementedError
    @abstractmethod
    def getCalibration(self):
        raise NotImplementedError
    @abstractmethod
    def getNContentLayers(self):
        raise NotImplementedError
    @abstractmethod
    def getNStyleLayers(self):
        raise NotImplementedError

# as particular example vgg19 architecture with imagenet weights was selected as NST model
class VGG19ForNST(ModelInterfaceForNST):
    def __init__(self):
        self.content_layers = ['block5_conv2'] 
        self.style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
        # defines calibration from image net. Those values are to subtracted in order to centralize mean of pixels values for each channel
        self.averagesOverChannel = [103.939, 116.779, 123.68]
        self.n_content = len(self.content_layers)
        self.n_style = len(self.style_layers)
    def preprocess_input(self, img):
        return tf.keras.applications.vgg19.preprocess_input(img)
    def get_model(self):
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
        # in the given implementation of NST the weights are not changed, that's why they are not trainable
        vgg.trainable = False
        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_outputs = [vgg.get_layer(name).output for name in self.content_layers]
        model_outputs = style_outputs + content_outputs
        # Vgg19 is reduced up to "block5_conv2" inclusively. Only convolution layers are considered in this implementation
        return models.Model(vgg.input, model_outputs)
    
    def getCalibration(self):
        return self.averagesOverChannel
    def getNContentLayers(self):
        return self.n_content
    def getNStyleLayers(self):
        return self.n_style
