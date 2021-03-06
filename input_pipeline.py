''' code from https://github.com/taki0112/Densenet-Tensorflow'''

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import os
import random
import imageio
import glob
from PIL import Image
import skimage.transform


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

                
def load_checkpoint(sess,checkpoint_dir, iteration=None,model_name='NET.model'):
        print(" [*] Reading checkpoints...")
        print(type(checkpoint_dir) ,checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and iteration:
            # Restores dump of given iteration
            ckpt_name = model_name + '-' + str(iteration)
        elif ckpt and ckpt.model_checkpoint_path:
            # Restores most recent dump
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Testing, but %s not found" % checkpoint_dir)

        ckpt_file = os.path.join(checkpoint_dir, ckpt_name)
        print('Reading variables to be restored from ' + ckpt_file)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        saver.restore(sess, ckpt_file)
        return ckpt_name

