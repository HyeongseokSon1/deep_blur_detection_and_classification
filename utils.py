import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
from config import config, log_config
from skimage import feature
from skimage import color
from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageChops
import cv2
import math
from tensorflow.python.ops import array_ops
import random


import scipy
import numpy as np

def get_imgs_RGB_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def get_imgs_GRAY_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    image = scipy.misc.imread(path + file_name, mode='P')/255
    return np.expand_dims(np.asarray(image), 3)

def get_imgs_GRAY_fn_cv(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    image = cv2.imread(path + file_name, cv2.IMREAD_GRAYSCALE)
    return np.expand_dims(np.asarray(image), 3)



def crop_sub_img_and_classification_fn_aug(data):

    dx = config.TRAIN.width
    dy = config.TRAIN.height
    image, mask = data
    #print "image shape", image.shape
    #print "mask shape", mask.shape

    image_h, image_w = np.asarray(image).shape[0:2]

    x = np.random.randint(0, image_w - dx - 1)
    y = np.random.randint(0, image_h - dy - 1)
    cropped_image = image[y: y+dy, x : x+dx, :]
    #print "hi2"
    cropped_mask  = mask[y: y + dy, x: x + dx, :]
    cropped_mask = np.concatenate((cropped_mask, cropped_mask, cropped_mask), axis=2)

    rotation_list = [0,90,180,270]
    flip_list = [0,1]
    rotation= random.choice(rotation_list)
    flip= random.choice(flip_list)

    if(flip ==1):
        cropped_image = cv2.flip(cropped_image,0)
        cropped_mask = cv2.flip(cropped_mask,0)
    rotation_matrix = cv2.getRotationMatrix2D((dy/2, dx/2), rotation, 1)
    cropped_image = cv2.warpAffine(cropped_image, rotation_matrix,(dy, dx))

    cropped_mask = cv2.warpAffine(cropped_mask, rotation_matrix,(dy, dx))
    #scipy.misc.imsave('samples/input_image.png', cropped_image)
    #scipy.misc.imsave('samples/input_mask.png', cropped_mask[:,:,0])

    #print edge.shape

    #print "hi3"
    # cropped_mask = np.expand_dims(cropped_mask, axis=3)


    #score = (np.sum(cropped_mask) / (dx * dy))

    #print "image shape", cropped_image.shape
   # print "mask shape", cropped_mask.shape

    #cropped_image = np.zeros((192,192,3))
    #cropped_mask = np.zeros((192,192,1))


    return cropped_image/(255.), cropped_mask


def crop_sub_img_and_classification_fn(data):

    dx = config.TRAIN.width
    dy = config.TRAIN.height
    image, mask = data
    #print "image shape", image.shape
    #print "mask shape", mask.shape

    image_h, image_w = np.asarray(image).shape[0:2]

    x = np.random.randint(0, image_w - dx - 1)
    y = np.random.randint(0, image_h - dy - 1)
    cropped_image = image[y: y+dy, x : x+dx, :]
    #print "hi2"
    cropped_mask  = mask[y: y + dy, x: x + dx, :]
    cropped_mask = np.concatenate((cropped_mask, cropped_mask, cropped_mask), axis=2)

    #print edge.shape

    #print "hi3"
    # cropped_mask = np.expand_dims(cropped_mask, axis=3)


    #score = (np.sum(cropped_mask) / (dx * dy))

    #print "image shape", cropped_image.shape
   # print "mask shape", cropped_mask.shape

    #cropped_image = np.zeros((192,192,3))
    #cropped_mask = np.zeros((192,192,1))


    return cropped_image/(255.), cropped_mask




