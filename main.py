import os
import scipy.misc
import numpy as np
from model import LAYOUTGAN
from utils import pp, show_all_variables
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 50, "Epoch to train [25]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [mnist, doc, clipart, tangram]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    layoutgan = LAYOUTGAN(sess,
      batch_size=FLAGS.batch_size,
      sample_num=FLAGS.batch_size,
      dataset_name=FLAGS.dataset,
      checkpoint_dir=FLAGS.checkpoint_dir,
      sample_dir=FLAGS.sample_dir)

    show_all_variables()

    if FLAGS.train:
      layoutgan.train(FLAGS)
    else:
      if not layoutgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")

if __name__ == '__main__':
  tf.app.run()
