# coding=utf-8
import tensorlayer as tl
import numpy as np
import math
from config import config, log_config
from utils import *
from model import *
import matplotlib
import datetime
import time
import cv2

import os

def read_all_imgs(img_list, path='', n_threads=32, mode = 'RGB'):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        if mode is 'RGB':
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_RGB_fn, path=path)
        elif mode is 'GRAY':
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_GRAY_fn, path=path)
        elif mode is 'GRAY_cv':
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_GRAY_fn_cv, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs

def blurmap_3classes(index):
    print "Blurmap Generation"
    
    date = datetime.datetime.now().strftime("%y.%m.%d")
    save_dir_sample = './output'
    tl.files.exists_or_mkdir(save_dir_sample)

    #Put the input path!
    sharp_path = './input'
    test_sharp_img_list = os.listdir(sharp_path)
    test_sharp_img_list.sort()


    flag=0
    i=0


    for image in test_sharp_img_list:
        if(i>=index and i<index+100):
            print i
            if (image.find('.jpg') & image.find('.png') & image.find('.JPG')&image.find('.PNG')) is not -1:

                sharp = os.path.join(sharp_path, image)
                sharp_image = Image.open(sharp)
                sharp_image.load()

                sharp_image = np.asarray(sharp_image, dtype="float32")
               
                if(len(sharp_image.shape)<3):
                    sharp_image= np.expand_dims(np.asarray(sharp_image), 3)
                    sharp_image=np.concatenate([sharp_image, sharp_image, sharp_image],axis=2)

                if (sharp_image.shape[2] ==4):
                    print sharp_image.shape
                    sharp_image = np.expand_dims(np.asarray(sharp_image), 3)

                    print sharp_image.shape
                    sharp_image = np.concatenate((sharp_image[:,:,0],sharp_image[:,:,1],sharp_image[:,:,2]),axis=2)

                print sharp_image.shape

                image_h, image_w =sharp_image.shape[0:2]
                print image_h, image_w

                test_image = sharp_image[0: image_h-(image_h%16), 0: 0 + image_w-(image_w%16), :]/(255.)

                # Model
                patches_blurred = tf.placeholder('float32', [1, test_image.shape[0], test_image.shape[1], 3], name='input_patches')
                if flag==0:
                    reuse =False
                else:
                    reuse =True

                start_time = time.time()

                with tf.variable_scope('Unified') as scope:
                    with tf.variable_scope('VGG') as scope3:
                        n, f0, f0_1, f1_2, f2_3, hrg, wrg = VGG19_pretrained(patches_blurred, reuse=reuse,scope=scope3)
                        #tl.visualize.draw_weights(n.all_params[0].eval(), second=10, saveable=True, name='weight_of_1st_layer', fig_idx=2012)

                    with tf.variable_scope('UNet') as scope1:
                        output,m1,m2,m3= Decoder_Network_classification(n, f0, f0_1, f1_2, f2_3 ,hrg,wrg, reuse = reuse, scope = scope1)
                   
                    output_map = tf.nn.softmax(output.outputs)
                    output_map1 = tf.nn.softmax(m1.outputs)
                    output_map2 = tf.nn.softmax(m2.outputs)
                    output_map3 = tf.nn.softmax(m3.outputs)

                a_vars = tl.layers.get_variables_with_name('Unified', False, True)

                saver = tf.train.Saver()
                sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
                tl.layers.initialize_global_variables(sess)

                # Load checkpoint
                saver.restore(sess, "./model/final_model.ckpt")



                start_time = time.time()
                blur_map,o1,o2,o3 = sess.run([output_map,output_map1,output_map2,output_map3],{patches_blurred: np.expand_dims(
                    (test_image ), axis=0)})
                blur_map = np.squeeze(blur_map )
                o1= np.squeeze(o1)
                o2 = np.squeeze(o2)
                o3 = np.squeeze(o3)

        

                if ".jpg" in image:
                    image.replace(".jpg", ".png")
                    cv2.imwrite(save_dir_sample +  '/'+ image.replace(".jpg", ".png"), blur_map*255)
                    
                if ".JPG" in image:
                    image.replace(".JPG", ".png")
                    cv2.imwrite(save_dir_sample +  '/'+ image.replace(".JPG", ".png"), blur_map*255)
                if ".PNG" in image:
                    image.replace(".jpg", ".png")
                    
                    cv2.imwrite(save_dir_sample +  '/'+ image.replace(".jpg", ".png"), blur_map*255)
                

                sess.close()
                flag=1


                print("5.--- %s seconds ---" % (time.time() - start_time))
                start_time = time.time()
                if(i==index+101-1):
                    return 0

        i = i + 1

    return 0




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--index', type=int, default='0', help='index range 50')
    args = parser.parse_args()    
    blurmap_3classes(args.index) # test_code
       