#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *



def UNet(t_image, is_train=False, reuse=False, scope = "UNet"):
    w_init1 = tf.random_normal_initializer(stddev=0.02)
    w_init2 = tf.random_normal_initializer(stddev=0.01)
    w_init3 = tf.random_normal_initializer(stddev=0.005)
    w_init4 = tf.random_normal_initializer(stddev=0.002)
    b_init = None # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    hrg = t_image.get_shape()[1]
    wrg = t_image.get_shape()[2]
    with tf.variable_scope(scope, reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        # n_init = InputLayer(t_image, name='in2')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init1, name='f0/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='f0/b')
        f0 = n
        n = Conv2d(n, 64, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init2, name='d1/c1')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='d1/b1')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init2, name='d1/c2')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='d1/b2')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init2, name='d1/c3')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='d1/b3')
        f1_2 = n
        n = Conv2d(n, 256, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init3, name='d2/c1')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='d2/b1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init3, name='d2/c2')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='d2/b2')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init3, name='d2/c3')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='d2/b3')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init3, name='d2/c4')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='d2/b4')
        f2_3 = n
        n = Conv2d(n, 512, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init4, name='d3/c1')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='d3/b1')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init4, name='d3/c2')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='d3/b2')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init4, name='d3/c3')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='d3/b3')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init4, name='d3/c4')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='d3/b4')

        n = DeConv2d(n, 256, (3, 3), (hrg/4, wrg/4), (2, 2), act=None, padding='SAME', W_init=w_init3, name='u3/d')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u3/b')
        n = ElementwiseLayer([n, f2_3], tf.add, name='s4')
        # n = InputLayer(tf.nn.relu(n.outputs), name='relu4')
        n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init3, name='u3/c1')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init3, name='u3/c2')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b2')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init3, name='u3/c3')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b3')

        n = DeConv2d(n, 128, (3, 3), (hrg/2, wrg/2), (2, 2), act=None, padding='SAME', W_init=w_init2, name='u2/d')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u2/b')
        n = ElementwiseLayer([n, f1_2], tf.add, name='s3')
        # n = InputLayer(tf.nn.relu(n.outputs), name='relu3')
        n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init2, name='u2/c1')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u2/b1')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init2, name='u2/c2')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u2/b2')

        n = DeConv2d(n, 64, (3, 3), (hrg, wrg), (2, 2), act=None, padding='SAME', W_init=w_init1, name='u1/d')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u1/b')
        n = ElementwiseLayer([n, f0], tf.add, name='s2')
        n.outputs = tf.nn.relu(n.outputs)
        #n = InputLayer(tf.nn.relu(n.outputs), name='relu2')
        n = Conv2d(n, 15, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init1, name='u1/c1')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u1/b1')
        n = Conv2d(n, 1, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init1, name='u1/c2')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u1/b2')
        # n = ElementwiseLayer([n, n_init], tf.add, name='s1')
        n2 = n
        n.outputs = tf.nn.sigmoid(n.outputs)
        # n2 = InputLayer(tf.nn.sigmoid(n.outputs), name='sigmoid')

        return n, n2

def UNet_(t_image, is_train=False, reuse=False, scope = "UNet"):
    w_init1 = tf.random_normal_initializer(stddev=0.02)
    w_init2 = tf.random_normal_initializer(stddev=0.01)
    w_init3 = tf.random_normal_initializer(stddev=0.005)
    w_init4 = tf.random_normal_initializer(stddev=0.002)
    b_init = None # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    hrg = t_image.get_shape()[1]
    wrg = t_image.get_shape()[2]
    with tf.variable_scope(scope, reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        # n_init = InputLayer(t_image, name='in2')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init1, name='f0/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='f0/b')
        f0 = n
        n = Conv2d(n, 64, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init2, name='d1/c1')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='d1/b1')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init2, name='d1/c2')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='d1/b2')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init2, name='d1/c3')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='d1/b3')
        f1_2 = n
        n = Conv2d(n, 256, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init3, name='d2/c1')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='d2/b1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init3, name='d2/c2')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='d2/b2')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init3, name='d2/c3')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='d2/b3')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init3, name='d2/c4')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='d2/b4')
        f2_3 = n
        n = Conv2d(n, 512, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init4, name='d3/c1')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='d3/b1')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init4, name='d3/c2')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='d3/b2')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init4, name='d3/c3')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='d3/b3')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init4, name='d3/c4')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='d3/b4')

        n = DeConv2d(n, 256, (3, 3), (hrg/4, wrg/4), (2, 2), act=None, padding='SAME', W_init=w_init3, name='u3/d')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u3/b')
        n = ElementwiseLayer([n, f2_3], tf.add, name='s4')
        # n = InputLayer(tf.nn.relu(n.outputs), name='relu4')
        n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init3, name='u3/c1')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init3, name='u3/c2')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b2')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init3, name='u3/c3')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b3')

        n = DeConv2d(n, 128, (3, 3), (hrg/2, wrg/2), (2, 2), act=None, padding='SAME', W_init=w_init2, name='u2/d')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u2/b')
        n = ElementwiseLayer([n, f1_2], tf.add, name='s3')
        # n = InputLayer(tf.nn.relu(n.outputs), name='relu3')
        n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init2, name='u2/c1')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u2/b1')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init2, name='u2/c2')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u2/b2')

        n = DeConv2d(n, 64, (3, 3), (hrg, wrg), (2, 2), act=None, padding='SAME', W_init=w_init1, name='u1/d')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u1/b')
        n = ElementwiseLayer([n, f0], tf.add, name='s2')
        n.outputs = tf.nn.relu(n.outputs)
        #n = InputLayer(tf.nn.relu(n.outputs), name='relu2')
        n = Conv2d(n, 15, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init1, name='u1/c1')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u1/b1')
        n = Conv2d(n, 1, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init1, name='u1/c2')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u1/b2')
        # n = ElementwiseLayer([n, n_init], tf.add, name='s1')
        n2=n
        n2.outputs = tf.nn.sigmoid(n.outputs)
        # n2 = InputLayer(tf.nn.sigmoid(n.outputs), name='sigmoid')

        return n





def VGG19_pretrained(t_image,reuse = False,scope="VGG"):
    hrg = t_image.get_shape()[1]
    wrg = t_image.get_shape()[2]
    VGG_MEAN = [103.939, 116.779, 123.68]
    """
        Build the VGG 19 Model
        Parameters
        -----------
        rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
        """
    print("build model started")
    rgb_scaled = t_image * 255.0
    # Convert RGB to BGR
    if tf.__version__ <= '0.11':
        red, green, blue = tf.split(3, 3, rgb_scaled)
    else:  # TF 1.0
        print(rgb_scaled)
        red, green, blue = tf.split(rgb_scaled, 3, 3)

    if tf.__version__ <= '0.11':
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
    else:
        bgr = tf.concat(
            [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], axis=3)
    with tf.variable_scope(scope, reuse=reuse) as vs:

        # input layer
        net_in = InputLayer(bgr, name='input')
        # conv1
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
        f0 = network
        #n.outputs= tf.nn.relu(n.outputs)
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')

        # conv2
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
        f0_1 = network
        #n.outputs = tf.nn.relu(n.outputs)
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')

        # conv3
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')
        f1_2 = network
        #n.outputs = tf.nn.relu(n.outputs)
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
        # conv4
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4')
        f2_3 = network
        #n.outputs=tf.nn.relu(n.outputs)
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')
        # conv5
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
        n = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_4')

    return n, f0, f0_1,f1_2,f2_3 ,hrg,wrg



def VGG19_finetuning(t_image,reuse = False,scope="VGG"):
    hrg = t_image.get_shape()[1]
    wrg = t_image.get_shape()[2]
    VGG_MEAN = [103.939, 116.779, 123.68]
    """
        Build the VGG 19 Model
        Parameters
        -----------
        rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
        """
    print("build model started")
    rgb_scaled = t_image * 255.0
    # Convert RGB to BGR
    if tf.__version__ <= '0.11':
        red, green, blue = tf.split(3, 3, rgb_scaled)
    else:  # TF 1.0
        print(rgb_scaled)
        red, green, blue = tf.split(rgb_scaled, 3, 3)

    if tf.__version__ <= '0.11':
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
    else:
        bgr = tf.concat(
            [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], axis=3)
    with tf.variable_scope(scope, reuse=reuse) as vs:

        # input layer
        net_in = InputLayer(bgr, name='input')
        # conv1
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
        f0 = network
        #n.outputs= tf.nn.relu(n.outputs)
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')

        # conv2
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
        f0_1 = network
        #n.outputs = tf.nn.relu(n.outputs)
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')

        # conv3
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')
        f1_2 = network
        #n.outputs = tf.nn.relu(n.outputs)
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
        # conv4
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4')
        f2_3 = network
        #n.outputs=tf.nn.relu(n.outputs)
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')
        # conv5
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
        n = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_4')


    return n, f0, f0_1,f1_2,f2_3 ,hrg,wrg









def Decoder_Network_classification(n,f0,f1_2,f2_3,f3_4,hrg,wrg, reuse=False, scope = "UNet"):
    w_init1 = tf.contrib.layers.xavier_initializer()
    w_init2 = tf.contrib.layers.xavier_initializer()
    w_init3 = tf.contrib.layers.xavier_initializer()
    w_init4 = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope(scope, reuse=reuse) as vs:

        #this bug..... w_init3->w_init4
        n = DeConv2d(n, 512, (3, 3), (hrg / 8, wrg / 8), (2, 2), act=None, padding='SAME', W_init=w_init4,
                     name='u4/d')
        # n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u3/b')
        # n.outputs = tf.nn.relu(n.outputs)
        f3_4 = Conv2d(f3_4, 512, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init4, name='f3_4/c1')
        n = ElementwiseLayer([n, f3_4], tf.add, name='s5')
        # n = InputLayer(tf.nn.relu(n.outputs), name='relu4')
        n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init4, name='u34/c1')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init4, name='u4/c2')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b2')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init4, name='u4/c3')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b3')
        # n.outputs = tf.nn.relu(n.outputs)
        n_m3 = Conv2d(n, 3, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init4, name='u4/loss3')

        n = DeConv2d(n, 256, (3, 3), (hrg / 4, wrg / 4), (2, 2), act=None, padding='SAME', W_init=w_init3,
                     name='u3/d')
        # n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u3/b')
        f2_3 = Conv2d(f2_3, 256, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init3, name='f2_3/c1')
        n = ElementwiseLayer([n, f2_3], tf.add, name='s4')
        # n = InputLayer(tf.nn.relu(n.outputs), name='relu4')
        n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init3, name='u3/c1')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init3, name='u3/c2')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b2')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init3, name='u3/c3')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b3')
        # n.outputs = tf.nn.relu(n.outputs)
        n_m2 = Conv2d(n,3, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init3, name='u3/loss2')

        n = DeConv2d(n, 128, (3, 3), (hrg / 2, wrg / 2), (2, 2), act=None, padding='SAME', W_init=w_init2,
                     name='u2/d')
        # n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u2/b')
        # n.outputs = tf.nn.relu(n.outputs)
        f1_2 = Conv2d(f1_2, 128, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init2, name='f1_2/c1')
        n = ElementwiseLayer([n, f1_2], tf.add, name='s3')
        # n = InputLayer(tf.nn.relu(n.outputs), name='relu3')
        n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init2, name='u2/c1')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u2/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init2, name='u2/c2')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u2/b2')
        # n.outputs = tf.nn.relu(n.outputs)
        n_m1 = Conv2d(n, 3, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init2, name='u2/loss1')

        n = DeConv2d(n, 64, (3, 3), (hrg, wrg), (2, 2), act=None, padding='SAME', W_init=w_init1, name='u1/d')
        # n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u1/b')
        # n.outputs = tf.nn.relu(n.outputs)
        f0 = Conv2d(f0, 64, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init1, name='f0/c1')
        n = ElementwiseLayer([n, f0], tf.add, name='s2')
        n.outputs = tf.nn.relu(n.outputs)
        # n = InputLayer(tf.nn.relu(n.outputs), name='relu2')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init1, name='u1/c1')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u1/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init1, name='u1/c2')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u1/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        #n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init1, name='u1/c3')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u1/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 3, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init1, name='u1/c5')
        # n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u1/b2')
        # n.outputs = tf.nn.relu(n.outputs)
        # n = ElementwiseLayer([n, n_init], tf.add, name='s1')
        #n.outputs = tf.nn.sigmoid(n.outputs)  # -> this is bug??
        #n2 = n

        # n2 = InputLayer(tf.nn.sigmoid(n.outputs), name='sigmoid')

    return n, n_m1, n_m2, n_m3
'''
def Decoder_Network_classification(n,f0,f1_2,f2_3,f3_4,hrg,wrg, reuse=False, scope = "UNet"):
    w_init1 = tf.contrib.layers.xavier_initializer()
    w_init2 = tf.contrib.layers.xavier_initializer()
    w_init3 = tf.contrib.layers.xavier_initializer()
    w_init4 = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope(scope, reuse=reuse) as vs:

        #this bug..... w_init3->w_init4
        n = DeConv2d(n, 512, (3, 3), (hrg / 8, wrg / 8), (2, 2), act=None, padding='SAME', W_init=w_init4,
                     name='u4/d')
        # n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u3/b')
        # n.outputs = tf.nn.relu(n.outputs)
        f3_4 = Conv2d(f3_4, 512, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init4, name='f3_4/c1')
        n = ElementwiseLayer([n, f3_4], tf.add, name='s5')
        # n = InputLayer(tf.nn.relu(n.outputs), name='relu4')
        n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init4, name='u34/c1')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init4, name='u4/c2')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b2')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init4, name='u4/c3')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b3')
        # n.outputs = tf.nn.relu(n.outputs)
        n_m3 = Conv2d(n, 3, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init4, name='u4/loss3')

        n = DeConv2d(n, 256, (3, 3), (hrg / 4, wrg / 4), (2, 2), act=None, padding='SAME', W_init=w_init3,
                     name='u3/d')
        # n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u3/b')
        f2_3 = Conv2d(f2_3, 256, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init3, name='f2_3/c1')
        n = ElementwiseLayer([n, f2_3], tf.add, name='s4')
        # n = InputLayer(tf.nn.relu(n.outputs), name='relu4')
        n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init3, name='u3/c1')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init3, name='u3/c2')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b2')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init3, name='u3/c3')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b3')
        # n.outputs = tf.nn.relu(n.outputs)
        n_m2 = Conv2d(n,3, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init3, name='u3/loss2')

        n = DeConv2d(n, 128, (3, 3), (hrg / 2, wrg / 2), (2, 2), act=None, padding='SAME', W_init=w_init2,
                     name='u2/d')
        # n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u2/b')
        # n.outputs = tf.nn.relu(n.outputs)
        f1_2 = Conv2d(f1_2, 128, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init2, name='f1_2/c1')
        n = ElementwiseLayer([n, f1_2], tf.add, name='s3')
        # n = InputLayer(tf.nn.relu(n.outputs), name='relu3')
        n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init2, name='u2/c1')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u2/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init2, name='u2/c2')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u2/b2')
        # n.outputs = tf.nn.relu(n.outputs)
        n_m1 = Conv2d(n, 3, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init2, name='u2/loss1')

        n = DeConv2d(n, 64, (3, 3), (hrg, wrg), (2, 2), act=None, padding='SAME', W_init=w_init1, name='u1/d')
        # n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u1/b')
        # n.outputs = tf.nn.relu(n.outputs)
        f0 = Conv2d(f0, 64, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init1, name='f0/c1')
        n = ElementwiseLayer([n, f0], tf.add, name='s2')
        n.outputs = tf.nn.relu(n.outputs)
        # n = InputLayer(tf.nn.relu(n.outputs), name='relu2')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init1, name='u1/c1')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u1/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init1, name='u1/c2')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u1/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        #n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init1, name='u1/c3')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u1/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n, 3, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init1, name='u1/c5')
        # n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u1/b2')
        # n.outputs = tf.nn.relu(n.outputs)
        # n = ElementwiseLayer([n, n_init], tf.add, name='s1')
        #n.outputs = tf.nn.sigmoid(n.outputs)  # -> this is bug??
        #n2 = n

        # n2 = InputLayer(tf.nn.sigmoid(n.outputs), name='sigmoid')

    return n, n_m1, n_m2, n_m3
'''








