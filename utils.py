"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
import PIL
from PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def save_npy_img(images, size, image_path):
    palette=[]
    for i in range(256):
      palette.extend((i,i,i))
    palette[:3*21]=np.array([[0, 0, 0],
                            [128, 0, 0],
                            [0, 128, 0],
                            [0, 0, 128],
                            [128, 128, 0],
                            [128, 0, 128],
                            [0, 128, 128],
                            [128, 128, 128],
                            [64, 0, 0]], dtype='uint8').flatten()

    cls_map_all = np.zeros((images.shape[0], images.shape[1], images.shape[2], 3), dtype=np.uint8)

    for img_ind in range(images.shape[0]):
      binary_mask = images[img_ind, :, :, :]

      # Add background
      image_sum = np.sum(binary_mask, axis=-1)
      ind = np.where(image_sum==0)
      image_bk = np.zeros((binary_mask.shape[0], binary_mask.shape[1]), dtype=np.float32)
      image_bk[ind] = 1.0
      image_bk = np.reshape(image_bk, (binary_mask.shape[0], binary_mask.shape[1], 1))
      binary_mask = np.concatenate((image_bk, binary_mask), axis=-1)

      cls_map = np.zeros((binary_mask.shape[0], binary_mask.shape[1]), dtype=np.float32)
      cls_map = np.argmax(binary_mask, axis=2)

      cls_map_img = Image.fromarray(cls_map.astype(np.uint8))
      cls_map_img.putpalette(palette)
      cls_map_img = cls_map_img.convert('RGB')
      cls_map_all[img_ind, :, :, :] = np.array(cls_map_img)

    cls_map_all = np.squeeze(merge(cls_map_all, size))
    return scipy.misc.imsave(image_path, cls_map_all)


def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')


def image_manifold_size(num_images):
  manifold_h = int(np.floor(np.sqrt(num_images)))
  manifold_w = int(np.ceil(np.sqrt(num_images)))
  assert manifold_h * manifold_w == num_images
  return manifold_h, manifold_w

