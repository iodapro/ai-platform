from __future__ import division
import os
import scipy.misc
os.environ["PBR_VERSION"]='5.1.3'
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import time
from multiprocessing import Lock

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models 
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from sys import platform

def getSeparator():
    if platform == "linux" or platform == "linux2":
        return '/'
    elif platform == "darwin":
        return '/'
    elif platform == "win32":
        return '\\'

separator = getSeparator()

isTerminate = False
tf.enable_eager_execution()

content_layers = ['block5_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
                ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

mutex = Lock()

def load_img(path_to_img):
  max_dim = 512
  img = Image.open(path_to_img)
  long = max(img.size)
  scale = max_dim/long
  img = img.resize((int(round(img.size[0]*scale)), int(round(img.size[1]*scale))), Image.ANTIALIAS)
  
  img = kp_image.img_to_array(img)
  
  img = np.expand_dims(img, axis=0)
  return img

def imshow(img, title=None):
  out = np.squeeze(img, axis=0)
  out = out.astype('uint8')
  plt.imshow(out)
  if title is not None:
    plt.title(title)
  plt.imshow(out)
 
def load_and_process_img(path_to_img):
  img = load_img(path_to_img)
  img = tf.keras.applications.vgg19.preprocess_input(img)
  return img

def deprocess_img(processed_img):
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)
  assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
  if len(x.shape) != 3:
    raise ValueError("Invalid input to deprocessing image")
  
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  x = x[:, :, ::-1]

  x = np.clip(x, 0, 255).astype('uint8')
  return x

def get_model():
  vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  style_outputs = [vgg.get_layer(name).output for name in style_layers]
  content_outputs = [vgg.get_layer(name).output for name in content_layers]
  model_outputs = style_outputs + content_outputs
  return models.Model(vgg.input, model_outputs)

def get_content_loss(base_content, target):
  return tf.reduce_mean(tf.square(base_content - target))

def gram_matrix(input_tensor):
  channels = int(input_tensor.shape[-1])
  a = tf.reshape(input_tensor, [-1, channels])
  n = tf.shape(a)[0]
  gram = tf.matmul(a, a, transpose_a=True)
  return gram / tf.cast(n, tf.float32)

def get_style_loss(base_style, gram_target):
  height, width, channels = base_style.get_shape().as_list()
  gram_style = gram_matrix(base_style)
  return tf.reduce_mean(tf.square(gram_style - gram_target))

def get_feature_representations(model, content_path, style_paths):
    
  content_image = load_and_process_img(content_path)
  content_outputs = model(content_image)
  content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
  styles_features = []
  for style_path in style_paths:
    style_image = load_and_process_img(style_path)
  
    style_outputs = model(style_image)
  
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    styles_features.append(style_features)
  return styles_features, content_features

def compute_loss(model, loss_weights, init_image, gram_styles_features, content_features):

  style_weight, content_weight = loss_weights
  model_outputs = model(init_image)
  
  style_output_features = model_outputs[:num_style_layers]
  content_output_features = model_outputs[num_style_layers:]
  style_score = 0
  weight_per_style_layer = 1.0 / float(num_style_layers)
  for target_styles in gram_styles_features:
      for target_style, comb_style in zip(target_styles, style_output_features):
          style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
  style_score *= style_weight

  content_score = 0
  weight_per_content_layer = 1.0 / float(num_content_layers)
  for target_content, comb_content in zip(content_features, content_output_features):
    content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)
  content_score *= content_weight

  loss = style_score + content_score 
  return loss, style_score, content_score

def compute_grads(cfg):
  with tf.GradientTape() as tape: 
    all_loss = compute_loss(**cfg)
    
  total_loss = all_loss[0]
  return tape.gradient(total_loss, cfg['init_image']), all_loss

def run_style_transfer(content_path, 
                       style_paths,
                       outFile,
                       value,
                       num_iterations=1000,
                       content_weight=1e3, 
                       style_weight=1e-2): 
  model = get_model() 
  for layer in model.layers:
    layer.trainable = False
  styles_features, content_features = get_feature_representations(model, content_path, style_paths)
  gram_styles_features = [[gram_matrix(style_feature) for style_feature in style_features] for style_features in styles_features]
  init_image = load_and_process_img(content_path)
  init_image = tfe.Variable(init_image, dtype=tf.float32)
  opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)
  iter_count = 1

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
  
  norm_means = np.array([103.939, 116.779, 123.68])
  min_vals = -norm_means
  max_vals = 255 - norm_means   
  imgs = []
  for i in range(num_iterations):
    grads, all_loss = compute_grads(cfg)
    loss, style_score, content_score = all_loss
    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)
    end_time = time.time()
    with mutex:
        if isTerminate == True:
           return best_img, best_loss
        img2save = deprocess_img(init_image.numpy())
        value.value = i
        if( not os.path.exists('out') ):
            os.mkdir('out')
        scipy.misc.imsave('out'+separator+outFile+'.jpg', img2save)
    
    if loss < best_loss:
      best_loss = loss
      best_img = deprocess_img(init_image.numpy())
    print('Iteration: {}'.format(i))        
    print('Total loss: {:.4e}, ' 
          'style loss: {:.4e}, '
          'content loss: {:.4e}, '
          'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
      
  return best_img, best_loss

def NST(content_path, style_paths, outFile, value, nIters):
    best, best_loss= run_style_transfer(content_path, style_paths, outFile, value, num_iterations=nIters)
