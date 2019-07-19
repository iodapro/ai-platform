import os
import tensorflow as tf
from sys import platform

# general purpose fuction for keeping track projected difference between current result and expected output
def get_loss(result, target):
        return tf.reduce_mean(tf.square(result - target))
   
# matrix of scalar products or gram matrix
def gram_matrix(input_tensor):
        channels = int(input_tensor.shape[-1])
        a = tf.reshape(input_tensor, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)

# defining separator based on operating system
def getSeparator():
    if platform == "linux" or platform == "linux2" or platform == "darwin":
        return '/'
    elif platform == "win32":
        return '\\'

    
