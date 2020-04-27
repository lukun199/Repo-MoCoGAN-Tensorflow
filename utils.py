import tensorflow as tf
import cv2
import numpy as np


# layers definition-------------------------------------------#
def conv2d(name, tensor, ksize, out_dim, padding=None, stddev=0.02, stride=2, use_bias=True):
    if not padding:
        padding = 'SAME'

    with tf.variable_scope(name):
        w = tf.get_variable('w', [ksize, ksize, tensor.get_shape()[-1], out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=stddev))
        var = tf.nn.conv2d(tensor, w, [1, stride, stride, 1], padding=padding)

        if use_bias:
            b = tf.get_variable('b', [out_dim], 'float32', initializer=tf.constant_initializer(0.0))
            return tf.nn.bias_add(var, b)
        else:
            return var


def conv3d(name, tensor, ksize, out_dim, pad_dim, stddev=0.02, stride=2, depth=4, padding='VALID',use_bias=True):
    # [filter_depth, filter_height, filter_width, in_channels, out_channels]
    # [batch, in_depth, in_height, in_width, in_channels]
    with tf.variable_scope(name):
        w = tf.get_variable('w', [depth, ksize, ksize, tensor.get_shape()[-1], out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=stddev))
        tensor = tf.pad(tensor, [[0, 0], [0, 0], [pad_dim, pad_dim], [pad_dim, pad_dim], [0, 0]])
        var = tf.nn.conv3d(tensor, w, [1, 1, stride, stride, 1], padding=padding)

        if use_bias:
            b = tf.get_variable('b', [out_dim], 'float32', initializer=tf.constant_initializer(0.0))
            return tf.nn.bias_add(var, b)
        else:
            return var


def deconv2d(name,tensor, ksize, outshape, stddev=0.02, stride=2, padding='SAME',use_bias=True):
    """
    pay attention to the size of input and output features.
    SAME,S=2,IN=3,OUT=5
    SAME,S=3,IN=4,OUT=8
    """
    with tf.variable_scope(name):
        # the format of filter: [height, width, output_channels, input_channels] channel is the last dimension
        w = tf.get_variable('w', [ksize, ksize, outshape[-1], tensor.get_shape()[-1]], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=stddev))
        var = tf.nn.conv2d_transpose(tensor, w, outshape, strides=[1, stride, stride, 1], padding=padding)

        if use_bias:
            b = tf.get_variable('b', [outshape[-1]], 'float32', initializer=tf.constant_initializer(0.0))
            return tf.nn.bias_add(var, b)
        else:
            return var


def fully_connected(name,tensor, output_shape, use_bias=False):
    with tf.variable_scope(name):
        shape = tensor.get_shape().as_list()
        w = tf.get_variable('w', [shape[1], output_shape], dtype=tf.float32,  # have nothing to do with batch_size
                                    initializer=tf.random_normal_initializer(stddev=0.02))

        if use_bias:
            b = tf.get_variable('b', [output_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            return tf.matmul(tensor, w) + b
        else:
            return tf.matmul(tensor, w)


def batch_norm(name, tensor, is_training = True):
    # in most cases, we use tf.layers.batch_norm, instead of tf.nn (complicated) or tf.contrib (removed in tf 2.0)
    # ref https://www.cnblogs.com/a-little-v/p/9925420.html
    # and https://www.cnblogs.com/hrlnw/p/7227447.html
    with tf.variable_scope(name):
        return tf.layers.batch_normalization(tensor,
                                             epsilon=1e-5,
                                             momentum=0.9,
                                             training=is_training,
                                             scale=True
                                            )
"""
    return tf.layers.dense(input, units=out_num,
                         activation=activation_type,
                         kernel_initializer=self.initializer,
                         name='dense_2',
                         reuse=scope.reuse)
"""

def relu(name, tensor):
    with tf.variable_scope(name):
        return tf.nn.relu(tensor)

def lrelu(name,inputdata, alpha=0.2):
    with tf.variable_scope(name):
        return tf.nn.relu(inputdata) - alpha * tf.nn.relu(-inputdata)

def sigmoid(name, x):
    with tf.variable_scope(name):
        return tf.nn.sigmoid(x)

def tanh(name, x):
    with tf.variable_scope(name):
        return tf.nn.tanh(x)

def image_to_pn_1(image):
    # [0,255] -> [-1,+1]
    return image / 127.5 - 1.0

def pn_1_to_image(data):
    return (data + 1.0) * 127.5

# other tools----------------------------------------------c

def addnoise(coefficient, input, is_use=True):  # input is a tensor   followed the routine of MOCOGAN
    if is_use: return tf.add(input, coefficient * tf.random_normal(input.get_shape(), 0, 1, dtype=tf.float32))
    else: return input

def save_images(img, size, channel, path):
    # input: [batch,h,w,channel] in range(0,1)

    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h * size[0], w * size[1], channel))
    for idx, image in enumerate(img):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j * h:j * h + h, i * w:i * w + w, :] = image
    return cv2.imwrite(path, pn_1_to_image(merge_img))

