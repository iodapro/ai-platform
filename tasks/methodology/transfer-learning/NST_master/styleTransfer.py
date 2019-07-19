from __future__ import division
import os
import scipy.misc
os.environ["PBR_VERSION"]='5.1.3'
import numpy as np
from PIL import Image
import time
from multiprocessing import Lock

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models 
from tensorflow.python.keras import layers
import utils

# class responsible for handling nst for images
class NeuralStyleTransfer:
    def __init__(self, model):
        self.nst_model = model
        self.num_content_layers = self.nst_model.getNContentLayers()
        self.num_style_layers = self.nst_model.getNStyleLayers()
        self.averagesOverChannel = self.nst_model.getCalibration()
        self.mutex = Lock()
        
        self.isTerminate = False
        self.separator = utils.getSeparator()
        tf.enable_eager_execution()
    # resizing image changing dimension, mean values of pixels subtraction
    def load_img(self,path_to_img):
        max_dim = 512
        img = Image.open(path_to_img)
        long = max(img.size)
        scale = max_dim/long
        img = img.resize((int(round(img.size[0]*scale)), int(round(img.size[1]*scale))), Image.ANTIALIAS)
  
        img = kp_image.img_to_array(img)
  
        img = np.expand_dims(img, axis=0)
        return img
    def load_and_process_img(self, path_to_img):
        img = self.load_img(path_to_img)
        img = self.nst_model.preprocess_input(img)
        return img
    
    def deprocess_img(self, processed_img):
        x = processed_img.copy()
        if len(x.shape) == 4:
            x = np.squeeze(x, 0)
        assert len(x.shape) == 3, ("Input must be an image of "
                                "size [1, height, width, channel] or [height, width, channel]")
        if len(x.shape) != 3:
            raise ValueError("Error: Invalid input!")
  
        x[:, :, 0] += self.averagesOverChannel[0]
        x[:, :, 1] += self.averagesOverChannel[1]
        x[:, :, 2] += self.averagesOverChannel[2]
        x = x[:, :, ::-1]

        x = np.clip(x, 0, 255).astype('uint8')
        return x
    # calculation of losses for content image and style images correspondingly
    def get_content_loss(self, base_content, target):
        return utils.get_loss(base_content, target)

    def get_style_loss(self, base_style, gram_target):
        gram_style = utils.gram_matrix(base_style)
        return utils.get_loss(gram_style, gram_target)
    # calculation of output features for content and style images
    def get_feature_representations(self, model, content_path, style_paths):
    
        content_image = self.load_and_process_img(content_path)
        # last layer output of modified VGG19
        content_outputs = model(content_image)
        # last layer features of modified VGG19
        content_features = [content_layer[0] for content_layer in content_outputs[self.num_style_layers:]]
        styles_features = []
        # in case of style image features are to extracted for each layer, since style information is cartured on all levels of the neural net.
        for style_path in style_paths:
            style_image = self.load_and_process_img(style_path)
  
            style_outputs = model(style_image)
  
            style_features = [style_layer[0] for style_layer in style_outputs[:self.num_style_layers]]
            styles_features.append(style_features)
        return styles_features, content_features
    
    def compute_loss(self, model, loss_weights, init_image, gram_styles_features, content_features):

        style_weight, content_weight = loss_weights
        model_outputs = model(init_image)
  
        style_output_features = model_outputs[:self.num_style_layers]
        content_output_features = model_outputs[self.num_style_layers:]
        
        # calculation of style loss using gramm matrix formalism
        style_score = 0
        weight_per_style_layer = 1.0 / float(self.num_style_layers)
        for target_styles in gram_styles_features:
            for target_style, comb_style in zip(target_styles, style_output_features):
                style_score += weight_per_style_layer * self.get_style_loss(comb_style[0], target_style)
        style_score *= style_weight
        
        # calculation of content loss
        content_score = 0
        weight_per_content_layer = 1.0 / float(self.num_content_layers)
        for target_content, comb_content in zip(content_features, content_output_features):
            content_score += weight_per_content_layer* self.get_content_loss(comb_content[0], target_content)
        content_score *= content_weight

        # total loss
        loss = style_score + content_score 
        return loss, style_score, content_score

    # knowing loss function one can calculate gradients
    def compute_grads(self, cfg):
        with tf.GradientTape() as tape: 
            all_loss = self.compute_loss(**cfg)
    
            total_loss = all_loss[0]
        return tape.gradient(total_loss, cfg['init_image']), all_loss
    
    def run_style_transfer(self, content_path, 
                        style_paths,
                        outFile,
                        value,
                        num_iterations=1000,
                        content_weight=1e3, 
                        style_weight=1e-2): 
        model = self.nst_model.get_model() 
        for layer in model.layers:
            layer.trainable = False
            
        styles_features, content_features = self.get_feature_representations(model, content_path, style_paths)
        # gram matrixices are compared in modified and style images. 
        gram_styles_features = [[utils.gram_matrix(style_feature) for style_feature in style_features] for style_features in styles_features]
        
        init_image = self.load_and_process_img(content_path)
        init_image = tfe.Variable(init_image, dtype=tf.float32)
        # Adam gradient descent is performed for updating features of image rather than weights of VGG19.
        opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)
        best_loss, best_img = float('inf'), None
  
        loss_weights = (style_weight, content_weight)
        cfg = {
            'model': model,
            'loss_weights': loss_weights,
            'init_image': init_image,
            'gram_styles_features': gram_styles_features,
            'content_features': content_features
        }
        start_time = time.time()
        global_start = time.time()
  
        norm_means = np.array(self.averagesOverChannel)
        min_vals = -norm_means
        max_vals = 255 - norm_means   
        for i in range(num_iterations):
            # these gradients are calculated on each iteration and image updating is performed
            grads, all_loss = self.compute_grads(cfg)
            loss, style_score, content_score = all_loss
            opt.apply_gradients([(grads, init_image)])
            # control of pixel values is performed to avoid gradients explosion
            clipped = tf.clip_by_value(init_image, min_vals, max_vals)
            init_image.assign(clipped)
            end_time = time.time()
            # on each iteration mutex is captured and here signal for termination is given to the main thread using isTerminate
            with self.mutex:
                if self.isTerminate == True:
                    return best_img, best_loss
                img2save = self.deprocess_img(init_image.numpy())
                value.value = i
                if( not os.path.exists('out') ):
                    os.mkdir('out')
                scipy.misc.imsave('out'+self.separator+outFile+'.jpg', img2save)
    
            if loss < best_loss:
                best_loss = loss
                best_img = self.deprocess_img(init_image.numpy())
            print('Iteration: {}'.format(i))        
            print('Total loss: {:.4e}, ' 
                'style loss: {:.4e}, '
                'content loss: {:.4e}, '
                'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
      
        return best_img, best_loss

