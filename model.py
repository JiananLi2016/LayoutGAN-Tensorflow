from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import random

from ops import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"]="1"

class LAYOUTGAN(object):
  def __init__(self, sess, batch_size=64, sample_num=64, dataset_name='default', checkpoint_dir=None, sample_dir=None):
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch.
    """
    self.sess = sess

    self.batch_size = batch_size
    self.sample_num = sample_num
    self.dataset_name = dataset_name
    self.checkpoint_dir = checkpoint_dir

    self.d_bn0 = batch_norm(name='d_bn0')
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    self.g_bn0_0 = batch_norm(name='g_bn0_0')
    self.g_bn0_1 = batch_norm(name='g_bn0_1')
    self.g_bn0_2 = batch_norm(name='g_bn0_2')
    self.g_bn0_3 = batch_norm(name='g_bn0_3')
    self.g_bn1_0 = batch_norm(name='g_bn1_0')
    self.g_bn1_1 = batch_norm(name='g_bn1_1')
    self.g_bn1_2 = batch_norm(name='g_bn1_2')
    self.g_bn1_3 = batch_norm(name='g_bn1_3')

    self.g_bn_x0 = batch_norm(name='g_bn_x0')
    self.g_bn_x1 = batch_norm(name='g_bn_x1')
    self.g_bn_x2 = batch_norm(name='g_bn_x2')
    self.g_bn_x3 = batch_norm(name='g_bn_x3')


    self.data_pre = np.load('./data/pre_data_cls.npy')
    print "complete loading pre_dat.npy"
    print len(self.data_pre) 

    self.build_model()


  def build_model(self):

    self.inputs = tf.placeholder(tf.float32, [self.batch_size, 128, 2], name='real_images')
    self.z = tf.placeholder(tf.float32, [64, 128, 2], name='z')

    self.G = self.generator(self.z)
    self.D, self.D_logits = self.discriminator(self.inputs, reuse=False)
    self.sampler = self.sampler(self.z)
    self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
    
    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.d_loss = self.d_loss_real + self.d_loss_fake
    self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
    self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")

    self.saver = tf.train.Saver()

  def train(self, config):
    global_step = tf.Variable(0, trainable=False)
    epoch_step = len(self.data_pre) // config.batch_size    
    lr = tf.train.exponential_decay(0.00001, global_step, 20*epoch_step, 0.1, staircase=True)

    d_optim = tf.train.AdamOptimizer(lr, beta1=0.9).minimize(self.d_loss, var_list=self.d_vars, global_step=global_step)
    g_optim = tf.train.AdamOptimizer(lr, beta1=0.9).minimize(self.g_loss, var_list=self.g_vars)

    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()
    
    sample = self.data_pre[0:self.sample_num]
    sample_inputs = np.array(sample).astype(np.float32)
    sample_inputs = sample_inputs * 28.0 / 27.0 

    sample_z = np.random.normal(0.5, 0.15, (64, 128, 2))
  
    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
      np.random.shuffle(self.data_pre)
      batch_idxs = len(self.data_pre) // config.batch_size

      for idx in xrange(0, batch_idxs):
        batch = self.data_pre[idx*config.batch_size:(idx+1)*config.batch_size]
        batch_images = np.array(batch).astype(np.float32)
        batch_images = batch_images * 28.0 / 27.0 

        batch_z = np.random.normal(0.5, 0.15, (64, 128, 2))

        # Update D network
        _ = self.sess.run([d_optim], feed_dict={ self.inputs: batch_images, self.z: batch_z})

        # Update G network
        _ = self.sess.run([g_optim], feed_dict={ self.inputs: batch_images, self.z: batch_z})
        _ = self.sess.run([g_optim], feed_dict={ self.inputs: batch_images, self.z: batch_z})

        errD_fake = self.d_loss_fake.eval({ self.z: batch_z})
        errD_real = self.d_loss_real.eval({ self.inputs: batch_images})
        errG = self.g_loss.eval({self.inputs: batch_images, self.z: batch_z})

        counter += 1
        if np.mod(counter, 10) == 0: 
          print("Epoch: [%2d] [%4d/%4d] time: %4.4f, lr:%.8f, d_loss: %.4f, g_loss: %.4f" \
            % (epoch, idx, batch_idxs, time.time()-start_time, lr.eval(), errD_fake+errD_real, errG))

        if np.mod(counter, 200) == 1:
          samples, d_loss, g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss],
            feed_dict={self.z: sample_z, self.inputs: sample_inputs})

          samples = np.reshape(samples, (64, 128, 2))
          samples = 27.0 * samples
          img_all = np.zeros((64, 28, 28, 3), dtype=np.uint8)

          for img_ind in range(64):
            pointset = np.rint(samples[img_ind,:,:]).astype(np.int)
            pointset = pointset[~(pointset==0).all(1)]
           
            img = np.zeros((28,28), dtype=np.float32)
            img[pointset[:,0], pointset[:,1]] = 255
            img = Image.fromarray(img.astype('uint8'), 'L')

            img_all[img_ind, :, :, :] = np.array(img.convert('RGB'))

          img_all = np.squeeze(merge(img_all, image_manifold_size(samples.shape[0])))

          scipy.misc.imsave('./{}/train_{:02d}_{:04d}.jpg'.format(config.sample_dir, epoch, idx), img_all)
          print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 

        if np.mod(counter, 2000) == 0:
          self.save(config.checkpoint_dir, counter)


  def discriminator(self, image, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      layout = layout_point(image, 28, 28, name='layout')
      # For bbox layout generation
      # layout = layout_bbox(image, 60, 40, name='layout')

      net = lrelu(self.d_bn0(conv2d(layout, 32, k_h=5, k_w=5, d_h=2, d_w=2, padding='VALID', name='conv1')))
      net = lrelu(self.d_bn1(conv2d(net, 64, k_h=5, k_w=5, d_h=2, d_w=2, padding='VALID', name='conv2')))
      net = tf.reshape(net, [self.batch_size, -1])      
      net = lrelu(self.d_bn2(linear(net, 512, scope='fc2')))
      net = linear(net, 1, 'fc3')

    return tf.nn.sigmoid(net), net


  def generator(self, z):
    with tf.variable_scope("generator") as scope:
      gnet = tf.reshape(z, [64, 128, 1, 2])
      h0_0 = self.g_bn0_0(conv2d(gnet, 1024, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0_0'))
      h0_1 = tf.nn.relu(self.g_bn0_1(conv2d(gnet, 256, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0_1')))
      h0_2 = tf.nn.relu(self.g_bn0_2(conv2d(h0_1, 256, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0_2')))
      h0_3 = self.g_bn0_3(conv2d(h0_2, 1024, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0_3'))
      gnet = tf.nn.relu(tf.add(h0_0, h0_3))

      # For bbox layout generation
      # gnet = tf.reshape(z, [64, 9, 6, 4])
      # h0_0 = self.g_bn0_0(conv2d(gnet, 256, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0_0'))
      # h0_1 = tf.nn.relu(self.g_bn0_1(conv2d(gnet, 64, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0_1')))
      # h0_2 = tf.nn.relu(self.g_bn0_2(conv2d(h0_1, 64, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0_2')))
      # h0_3 = self.g_bn0_3(conv2d(h0_2, 256, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0_3'))
      # gnet = tf.nn.relu(tf.add(h0_0, h0_3))
      # gnet = tf.reshape(gnet, [64, 9, 1, 6*256])

      gnet = tf.reshape(gnet, [64, 128, 1, 1024])
      gnet = tf.nn.relu(self.g_bn_x1( tf.add(gnet, self.g_bn_x0(relation_nonLocal(gnet, name='g_non0')))))
      gnet = tf.nn.relu(self.g_bn_x3( tf.add(gnet, self.g_bn_x2(relation_nonLocal(gnet, name='g_non2')))))

      h1_0 = self.g_bn1_0(conv2d(gnet, 1024, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h1_0'))
      h1_1 = tf.nn.relu(self.g_bn1_1(conv2d(h1_0, 256, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h1_1')))
      h1_2 = tf.nn.relu(self.g_bn1_2(conv2d(h1_1, 256, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h1_2')))
      h1_3 = self.g_bn1_3(conv2d(h1_2, 1024, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h1_3'))
      gnet = tf.nn.relu(tf.add(h1_0, h1_3))

      # For bbox layout generation
      # May add more self-attention refinement steps

      bbox_pred = conv2d(gnet, 2, k_h=1, k_w=1, d_h=1, d_w=1, name='bbox_pred')
      bbox_pred = tf.sigmoid(tf.reshape(bbox_pred, [-1, 128, 2]))
      final_pred = bbox_pred

      # For bbox layout generation 
      # cls_score = conv2d(gnet, 6, k_h=1, k_w=1, d_h=1, d_w=1, name='cls_score')
      # cls_prob  = tf.sigmoid(tf.reshape(cls_score, [-1, 9, 6]))
      # final_pred = tf.concat([bbox_pred, cls_prob], axis=-1)

      return final_pred 


  def sampler(self, z):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()
      gnet = tf.reshape(z, [64, 128, 1, 2])
      h0_0 = self.g_bn0_0(conv2d(gnet, 1024, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0_0'), train=False)
      h0_1 = tf.nn.relu(self.g_bn0_1(conv2d(gnet, 256, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0_1'), train=False))
      h0_2 = tf.nn.relu(self.g_bn0_2(conv2d(h0_1, 256, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0_2'), train=False))
      h0_3 = self.g_bn0_3(conv2d(h0_2, 1024, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0_3'), train=False)
      gnet = tf.nn.relu(tf.add(h0_0, h0_3))

      gnet = tf.reshape(gnet, [64, 128, 1, 1024])
      gnet = tf.nn.relu(self.g_bn_x1( tf.add(gnet, self.g_bn_x0(relation_nonLocal(gnet, name='g_non0'), train=False)), train=False))
      gnet = tf.nn.relu(self.g_bn_x3( tf.add(gnet, self.g_bn_x2(relation_nonLocal(gnet, name='g_non2'), train=False)), train=False))

      h1_0 = self.g_bn1_0(conv2d(gnet, 1024, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h1_0'), train=False)
      h1_1 = tf.nn.relu(self.g_bn1_1(conv2d(h1_0, 256, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h1_1'), train=False))
      h1_2 = tf.nn.relu(self.g_bn1_2(conv2d(h1_1, 256, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h1_2'), train=False))
      h1_3 = self.g_bn1_3(conv2d(h1_2, 1024, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h1_3'), train=False)
      gnet = tf.nn.relu(tf.add(h1_0, h1_3))

      bbox_pred = conv2d(gnet, 2, k_h=1, k_w=1, d_h=1, d_w=1, name='bbox_pred')
      bbox_pred = tf.sigmoid(tf.reshape(bbox_pred, [-1, 128, 2]))
      final_pred = bbox_pred

      return final_pred 


  @property
  def model_dir(self):
    return "{}_{}".format(self.dataset_name, self.batch_size)
      
  def save(self, checkpoint_dir, step):
    model_name = "LAYOUTGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
