import math
import numpy as np 
import tensorflow as tf
from tensorflow.python.framework import ops
from utils import *

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, padding='SAME', name="conv2d"):
  with tf.variable_scope(name):
    if name == 'bbox_pred':
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                  initializer=tf.truncated_normal_initializer(0.0, stddev=0.001))
    else:
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                  initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv


def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias


def relation_nonLocal(input_, name="relation_nonLocal"):
  with tf.variable_scope(name):

    shape_org = input_.get_shape().as_list()
    N, H, W, C = shape_org[0], shape_org[1], shape_org[2], shape_org[3]
    # output_dim, d_k, d_g = C/2, C/2, C/2
    output_dim, d_k, d_g = C, C, C

    f_v = conv2d(input_, output_dim, k_h=1, k_w=1, d_h=1, d_w=1, name="f_v")
    f_k = conv2d(input_, d_k, k_h=1, k_w=1, d_h=1, d_w=1, name="f_k")
    f_q = conv2d(input_, d_k, k_h=1, k_w=1, d_h=1, d_w=1, name="f_q")

    f_k = tf.reshape(f_k, [N, H*W, d_k])
    f_q = tf.transpose(tf.reshape(f_q, [N, H*W, d_k]), perm=[0, 2, 1])
    w = tf.matmul(f_k, f_q)/(H*W)

    f_r = tf.matmul(tf.transpose(w, perm=[0, 2, 1]), tf.reshape(f_v, [N, H*W, output_dim]))
    f_r = tf.reshape(f_r, [N, H, W, output_dim])
    f_r = conv2d(f_r, C, k_h=1, k_w=1, d_h=1, d_w=1, name="f_r")

    return f_r 


def layout_point(final_pred, output_height, output_width, name="layout_point"):
  with tf.variable_scope(name):

    bbox_pred = tf.reshape(final_pred, [64, 128, 2])

    x_r = tf.reshape(tf.range(output_width, dtype=tf.float32), [1, output_width, 1, 1])
    x_r = tf.reshape(tf.tile(x_r, [1, 1, output_width, 1]), [1, output_width*output_width, 1, 1])
    x_r = tf.tile(x_r, [64, 1, 128, 1])

    y_r = tf.reshape(tf.range(output_height, dtype=tf.float32), [1, 1, output_height, 1])
    y_r = tf.reshape(tf.tile(y_r, [1, output_height, 1, 1]), [1, output_height*output_height, 1, 1])
    y_r = tf.tile(y_r, [64, 1, 128, 1])

    x_pred = tf.reshape(tf.slice(bbox_pred, [0, 0, 0], [-1, -1, 1]), [64, 1, 128, 1])
    x_pred = tf.tile(x_pred, [1, output_width*output_width, 1, 1])
    x_pred = (output_width-1.0) * x_pred

    y_pred = tf.reshape(tf.slice(bbox_pred, [0, 0, 1], [-1, -1, 1]), [64, 1, 128, 1])
    y_pred = tf.tile(y_pred, [1, output_height*output_height, 1, 1])
    y_pred = (output_height-1.0) * y_pred

    x_diff = tf.maximum(0.0, 1.0-tf.abs(x_r - x_pred))
    y_diff = tf.maximum(0.0, 1.0-tf.abs(y_r - y_pred))
    xy_diff = x_diff * y_diff

    xy_max = tf.nn.max_pool(xy_diff, ksize=[1, 1, 128, 1], strides=[1, 1, 1, 1], padding='VALID')
    xy_max = tf.reshape(xy_max, [64, output_height, output_width, 1])

    return xy_max 


# For bbox layout generation 
def layout_bbox(final_pred, output_height, output_width, name="layout_bbox"):
  with tf.variable_scope(name):

    final_pred = tf.reshape(final_pred, [64, 9, 10])
    bbox_reg = tf.slice(final_pred, [0, 0, 0], [-1, -1, 4])
    cls_prob = tf.slice(final_pred, [0, 0, 4], [-1, -1, 6])

    bbox_reg = tf.reshape(bbox_reg, [64, 9, 4])

    x_c = tf.slice(bbox_reg, [0, 0, 0], [-1, -1, 1]) * output_width
    y_c = tf.slice(bbox_reg, [0, 0, 1], [-1, -1, 1]) * output_height
    w   = tf.slice(bbox_reg, [0, 0, 2], [-1, -1, 1]) * output_width
    h   = tf.slice(bbox_reg, [0, 0, 3], [-1, -1, 1]) * output_height

    x1 = x_c - 0.5*w
    x2 = x_c + 0.5*w
    y1 = y_c - 0.5*h
    y2 = y_c + 0.5*h

    xt = tf.reshape(tf.range(output_width, dtype=tf.float32), [1, 1, 1, -1])
    xt = tf.reshape(tf.tile(xt, [64, 9, output_height, 1]), [64, 9, -1])

    yt = tf.reshape(tf.range(output_height, dtype=tf.float32), [1, 1, -1, 1])
    yt = tf.reshape(tf.tile(yt, [64, 9, 1, output_width]), [64, 9, -1])

    x1_diff = tf.reshape(xt-x1, [64, 9, output_height, output_width, 1])
    y1_diff = tf.reshape(yt-y1, [64, 9, output_height, output_width, 1])
    x2_diff = tf.reshape(x2-xt, [64, 9, output_height, output_width, 1])
    y2_diff = tf.reshape(y2-yt, [64, 9, output_height, output_width, 1])

    x1_line = tf.nn.relu(1.0 - tf.abs(x1_diff)) * tf.minimum(tf.nn.relu(y1_diff), 1.0) * tf.minimum(tf.nn.relu(y2_diff), 1.0)
    x2_line = tf.nn.relu(1.0 - tf.abs(x2_diff)) * tf.minimum(tf.nn.relu(y1_diff), 1.0) * tf.minimum(tf.nn.relu(y2_diff), 1.0)
    y1_line = tf.nn.relu(1.0 - tf.abs(y1_diff)) * tf.minimum(tf.nn.relu(x1_diff), 1.0) * tf.minimum(tf.nn.relu(x2_diff), 1.0)
    y2_line = tf.nn.relu(1.0 - tf.abs(y2_diff)) * tf.minimum(tf.nn.relu(x1_diff), 1.0) * tf.minimum(tf.nn.relu(x2_diff), 1.0)

    xy_max = tf.reduce_max(tf.concat([x1_line, x2_line, y1_line, y2_line], axis=-1), axis=-1, keep_dims=True)

    spatial_prob = tf.multiply(tf.tile(xy_max, [1, 1, 1, 1, 6]), tf.reshape(cls_prob, [64, 9, 1, 1, 6]))
    spatial_prob_max = tf.reduce_max(spatial_prob, axis=1, keep_dims=False)

    return spatial_prob_max
