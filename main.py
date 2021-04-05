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
# import cv2

import os

batch_size = config.TRAIN.batch_size
#batch_size = 78
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1

n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

h = config.TRAIN.height
w = config.TRAIN.width

ni = int(math.ceil(np.sqrt(batch_size)))

def read_all_imgs(img_list, path='', n_threads=32, mode = 'RGB'):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        if mode is 'RGB':
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_RGB_fn, path=path)
        elif mode is 'GRAY':
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_GRAY_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs


def train_with_CUHK():
    checkpoint_dir ="test_checkpoint/{}".format(tl.global_flag['mode'])  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)
    log_config(checkpoint_dir + '/config', config)

    save_dir_sample = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_sample)
    input_path = config.TRAIN.CUHK_blur_path  #for comparison with neurocomputing
    train_blur_img_list = sorted(tl.files.load_file_list(path=input_path, regx='(out_of_focus|motion).*.(jpg|JPG)', printable=False))
    train_mask_img_list=[]

    for str in train_blur_img_list:

        if ".jpg" in str:
            train_mask_img_list.append(str.replace(".jpg",".png"))
        else:
            train_mask_img_list.append(str.replace(".JPG", ".png"))


    gt_path = config.TRAIN.CUHK_gt_path
    print train_blur_img_list

    train_blur_imgs = read_all_imgs(train_blur_img_list, path=input_path, n_threads=batch_size ,mode='RGB')
    train_mask_imgs = read_all_imgs(train_mask_img_list, path=gt_path, n_threads=batch_size,mode='GRAY')
    train_edge_imgs = []
    for img in train_blur_imgs:
        edges = cv2.Canny(img, 100, 200)
        train_edge_imgs.append(edges)

    index= 0
    train_classification_mask= []
    #img_n = 0
    for img in train_mask_imgs:
        if(index<236):
            tmp_class = img
            tmp_classification = np.concatenate((img,img,img),axis = 2)
            tmp_class[np.where(tmp_classification[:,:,0]==0)] =0 #sharp 
            tmp_class[np.where(tmp_classification[:,:,0]>0)] =1 #defocus blur
        else:
            tmp_class = img
            tmp_classification = np.concatenate((img, img, img), axis=2)
            tmp_class[np.where(tmp_classification[:,:,0]==0)] =0 #sharp 
            tmp_class[np.where(tmp_classification[:,:,0]>0)] =2 #defocus blur

        train_classification_mask.append(tmp_class)
        index =index +1

    ### DEFINE MODEL ###
    patches_blurred = tf.placeholder('float32', [batch_size, h, w, 3], name = 'input_patches')
    labels_sigma = tf.placeholder('float32', [batch_size,h,w, 1], name = 'lables')
    classification_map= tf.placeholder('int32', [batch_size, h, w,1], name='labels')
    with tf.variable_scope('Unified'):
        with tf.variable_scope('VGG') as scope1:
            n, f0, f0_1, f1_2, f2_3 ,hrg,wrg= VGG19_pretrained(patches_blurred,reuse=False, scope=scope1)
        with tf.variable_scope('UNet') as scope2:
            net_regression,m1,m2,m3= Decoder_Network_classification(n, f0, f0_1, f1_2, f2_3 ,hrg,wrg, reuse = False, scope = scope2)

    ### DEFINE LOSS ###
    loss1 = tl.cost.cross_entropy((net_regression.outputs),  tf.squeeze( classification_map), name='loss1')
    loss2 = tl.cost.cross_entropy((m1.outputs),   tf.squeeze( tf.image.resize_images(classification_map, [128,128],method = tf.image.ResizeMethod.NEAREST_NEIGHBOR )),name ='loss2')
    loss3 = tl.cost.cross_entropy((m2.outputs),   tf.squeeze( tf.image.resize_images(classification_map, [64,64],method = tf.image.ResizeMethod.NEAREST_NEIGHBOR) ),name='loss3')
    loss4 = tl.cost.cross_entropy((m3.outputs), tf.squeeze( tf.image.resize_images(classification_map, [32,32],method = tf.image.ResizeMethod.NEAREST_NEIGHBOR )),name='loss4')
    out =(net_regression.outputs)
    loss = loss1 + loss2 + loss3 +loss4

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable = False)

    ### DEFINE OPTIMIZER ###
    vgg_vars = tl.layers.get_variables_with_name('VGG', True, True)  # ?
    t_vars = tl.layers.get_variables_with_name('UNet', True, True) #?
    a_vars = tl.layers.get_variables_with_name('Unified', False, True)  #
    var_list1 = vgg_vars
    var_list2 = t_vars
    opt1 = tf.train.AdamOptimizer(lr_v*0.1*0.1)
    opt2 = tf.train.AdamOptimizer(lr_v*0.1)
    grads = tf.gradients(loss, var_list1 + var_list2)
    grads1 = grads[:len(var_list1)]
    grads2 = grads[len(var_list1):]
    train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
    train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
    train_op = tf.group(train_op1, train_op2)


    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))
    print "initializing global variable..."
    tl.layers.initialize_global_variables(sess)
    print "initializing global variable...DONE"


    ### LOAD VGG ###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()
    #
    params = []
    count_layers = 0
    for val in sorted(npz.items()):
        if (count_layers < 16):
            W = np.asarray(val[1][0])
            b = np.asarray(val[1][1])
            print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
            params.extend([W, b])
        count_layers += 1

    tl.files.assign_params(sess, params, n)

    ### START TRAINING ###
    sess.run(tf.assign(lr_v, lr_init))
    global_step = 0
    new_lr_decay=1
    for epoch in range(0, n_epoch + 1):
        ## update learning rate
        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            #new_lr_decay = new_lr_decay * lr_decay
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_loss, n_iter = 0, 0
        new_batch_size = batch_size  #batchsize 50->40 + 10(augmented)

        #data suffle***
        suffle_index = np.arange(len(train_blur_imgs))
        np.random.shuffle(suffle_index)
        #print len(train_blur_imgs)
        #print suffle_index
        prev_train_blur_imgs =train_blur_imgs
        prev_train_classification_mask =train_classification_mask
        train_blur_imgs = []
        train_classification_mask =[]
        for i in range(0,len(suffle_index),1):
            train_blur_imgs.append(prev_train_blur_imgs[suffle_index[i]] )
            train_classification_mask.append(prev_train_classification_mask[suffle_index[i]] )

        for idx in range(0, len(train_blur_imgs), new_batch_size):
            step_time = time.time()

            augmentation_list = [0,1]
            augmentation= random.choice(augmentation_list)
            if(augmentation ==0):
                images_and_score = tl.prepro.threading_data([_ for _ in zip(train_blur_imgs[idx: idx + new_batch_size],train_classification_mask[idx: idx + new_batch_size])],fn=crop_sub_img_and_classification_fn)                
            elif (augmentation==1):
                images_and_score = tl.prepro.threading_data([_ for _ in zip(train_blur_imgs[idx: idx + new_batch_size],train_classification_mask[idx: idx + new_batch_size])],fn=crop_sub_img_and_classification_fn_aug)
                
            #print images_and_score.shape
            imlist, clist= images_and_score.transpose((1,0,2,3,4))
            #print clist.shape
            clist = clist[:, :, :, 0]
            #print clist.shape
            clist = np.expand_dims(clist, axis=3)

            #print imlist.shape, clist.shape
            err,l1,l2,l3,l4, _ ,outmap= sess.run([loss,loss1,loss2,loss3,loss4, train_op,out], {patches_blurred: imlist, classification_map: clist})

            outmap1 = np.squeeze(outmap[1,:,:,0])
            outmap2 = np.squeeze(outmap[1, :, :, 1])
            outmap3 = np.squeeze(outmap[1, :, :, 2])

            if(idx%100 ==0):
                scipy.misc.imsave(save_dir_sample + '/input_mask.png', np.squeeze(clist[1, :, :, 0]))
                scipy.misc.imsave(save_dir_sample + '/input.png', np.squeeze(imlist[1,:,:,:]))
                scipy.misc.imsave(save_dir_sample + '/im.png', outmap1)
                scipy.misc.imsave(save_dir_sample + '/im1.png', outmap2)
                scipy.misc.imsave(save_dir_sample + '/im2.png', outmap3)

            print("Epoch [%2d/%2d] %4d time: %4.4fs, err: %.6f, loss1: %.6f,loss2: %.6f,loss3: %.6f,loss4: %.6f" % (epoch, n_epoch, n_iter, time.time() - step_time, err,l1,l2,l3,l4))
            total_loss += err
            n_iter += 1
            global_step += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, total_err: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_loss/n_iter)
        print(log)

        ## save model
        if epoch % 200 == 0:
            tl.files.save_ckpt(sess=sess, mode_name='SA_net_{}.ckpt'.format(tl.global_flag['mode']), save_dir = checkpoint_dir, var_list = a_vars, global_step = global_step, printable = False)


def train_with_synthetic():
    checkpoint_dir ="test_checkpoint/{}".format(tl.global_flag['mode'])  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)
    log_config(checkpoint_dir + '/config', config)

    save_dir_sample = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_sample)
    input_path = config.TRAIN.synthetic_blur_path  
    train_blur_img_list = sorted(tl.files.load_file_list(path=input_path, regx='(out_of_focus|motion).*.(jpg|JPG)', printable=False))
    train_mask_img_list=[]

    for str in train_blur_img_list:

        if ".jpg" in str:
            train_mask_img_list.append(str.replace(".jpg",".png"))
        else:
            train_mask_img_list.append(str.replace(".JPG", ".png"))



    #augmented dataset read
    gt_path =config.TRAIN.synthetic_gt_path
    print train_mask_img_list

    train_blur_imgs = read_all_imgs(train_blur_img_list, path=input_path, n_threads=100 ,mode='RGB')
    train_mask_imgs = read_all_imgs(train_mask_img_list, path=gt_path, n_threads=100,mode='GRAY_cv')


    index= 0
    train_classification_mask= []
    #print train_mask_imgs
    #img_n = 0
    for img in train_mask_imgs:
        
        tmp_class = img
        tmp_classification = np.concatenate((img,img,img),axis = 2)

        tmp_class[np.where(tmp_classification[:,:,0]==0)] =0 #sharp 
        tmp_class[np.where(tmp_classification[:,:,0]==100)] =1 #motion blur
        tmp_class[np.where(tmp_classification[:,:,0]==200)] =2 #defocus blur
    
        train_classification_mask.append(tmp_class)
        index =index +1


    input_path2 = config.TRAIN.CUHK_blur_path  
    ori_train_blur_img_list = sorted(tl.files.load_file_list(path=input_path2, regx='(out_of_focus|motion).*.(jpg|JPG)', printable=False))
    ori_train_mask_img_list=[]

    for str in ori_train_blur_img_list:

        if ".jpg" in str:
            ori_train_mask_img_list.append(str.replace(".jpg",".png"))
        else:
            ori_train_mask_img_list.append(str.replace(".JPG", ".png"))



    #augmented dataset read
    gt_path2 = config.TRAIN.CUHK_gt_path
    print train_blur_img_list

    ori_train_blur_imgs = read_all_imgs(ori_train_blur_img_list, path=input_path2, n_threads=batch_size ,mode='RGB')
    ori_train_mask_imgs = read_all_imgs(ori_train_mask_img_list, path=gt_path2, n_threads=batch_size,mode='GRAY')
    train_edge_imgs = []


    index= 0
    ori_train_classification_mask= []
    #img_n = 0
    for img in ori_train_mask_imgs:
        if(index<236):
            tmp_class = img
            tmp_classification   = np.concatenate((img,img,img),axis = 2)

            tmp_class[np.where(tmp_classification[:,:,0]==0)] =0 #sharp 
            tmp_class[np.where(tmp_classification[:,:,0]>0)] =1 #defocus blur
        else:
            tmp_class = img
            tmp_classification = np.concatenate((img, img, img), axis=2)
            tmp_class[np.where(tmp_classification[:,:,0]==0)] =0 #sharp 
            tmp_class[np.where(tmp_classification[:,:,0]>0)] =2 #defocus blur

        ori_train_classification_mask.append(tmp_class)
        index =index +1
    train_mask_imgs=  train_classification_mask
    for i in range(10):
        train_blur_imgs = train_blur_imgs + ori_train_blur_imgs;
        train_mask_imgs = train_mask_imgs + ori_train_classification_mask;

    print len(train_blur_imgs), len(train_mask_imgs)


    ### DEFINE MODEL ###
    patches_blurred = tf.placeholder('float32', [batch_size, h, w, 3], name = 'input_patches')
    labels_sigma = tf.placeholder('float32', [batch_size,h,w, 1], name = 'lables')
    classification_map= tf.placeholder('int32', [batch_size, h, w,1], name='labels')
    #class_map = tf.placeholder('int32', [batch_size, h, w], name='classes')
    #attention_edge = tf.placeholder('float32', [batch_size, h, w, 1], name='attention')
    with tf.variable_scope('Unified'):
        with tf.variable_scope('VGG') as scope1:
            n, f0, f0_1, f1_2, f2_3 ,hrg,wrg= VGG19_pretrained(patches_blurred,reuse=False, scope=scope1)
        with tf.variable_scope('UNet') as scope2:
            net_regression,m1,m2,m3= Decoder_Network_classification(n, f0, f0_1, f1_2, f2_3 ,hrg,wrg, reuse = False, scope = scope2)

    ### DEFINE LOSS ###
    loss1 = tl.cost.cross_entropy((net_regression.outputs),  tf.squeeze( classification_map), name='loss1')
    loss2 = tl.cost.cross_entropy((m1.outputs),   tf.squeeze( tf.image.resize_images(classification_map, [128,128],method = tf.image.ResizeMethod.NEAREST_NEIGHBOR )),name ='loss2')
    loss3 = tl.cost.cross_entropy((m2.outputs),   tf.squeeze( tf.image.resize_images(classification_map, [64,64],method = tf.image.ResizeMethod.NEAREST_NEIGHBOR) ),name='loss3')
    loss4 = tl.cost.cross_entropy((m3.outputs), tf.squeeze( tf.image.resize_images(classification_map, [32,32],method = tf.image.ResizeMethod.NEAREST_NEIGHBOR )),name='loss4')
    out =(net_regression.outputs)
    loss = loss1 + loss2 +loss3 +loss4 

    #loss = tf.reduce_mean(tf.abs((net_regression.outputs + 1) - labels_sigma))

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable = False)

    ### DEFINE OPTIMIZER ###
    vgg_vars = tl.layers.get_variables_with_name('VGG', True, True)  # ?
    t_vars = tl.layers.get_variables_with_name('UNet', True, True) #?
    a_vars = tl.layers.get_variables_with_name('Unified', False, True)  #
    var_list1 = vgg_vars
    var_list2 = t_vars
    opt1 = tf.train.AdamOptimizer(lr_v*0.1*0.1)
    opt2 = tf.train.AdamOptimizer(lr_v*0.1)
    grads = tf.gradients(loss, var_list1 + var_list2)
    grads1 = grads[:len(var_list1)]
    grads2 = grads[len(var_list1):]
    train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
    train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
    train_op = tf.group(train_op1, train_op2)


    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))
    print "initializing global variable..."
    tl.layers.initialize_global_variables(sess)
    print "initializing global variable...DONE"

    ### initial checkpoint ###
    checkpoint_dir2 = "test_checkpoint/PG_CUHK/"
    tl.files.load_ckpt(sess=sess, mode_name='SA_net_PG_CUHK.ckpt', save_dir=checkpoint_dir2, var_list=a_vars, is_latest=True)


    ### START TRAINING ###
    sess.run(tf.assign(lr_v, lr_init))
    global_step = 0
    new_lr_decay=1
    prev_train_blur_imgs =train_blur_imgs
    prev_train_classification_mask =train_mask_imgs
    for epoch in range(0, n_epoch + 1):
        ## update learning rate
        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            #new_lr_decay = new_lr_decay * lr_decay
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_loss, n_iter = 0, 0
        new_batch_size = batch_size  #batchsize 50->40 + 10(augmented)

        #data suffle***
        suffle_index = np.arange(len(prev_train_blur_imgs))
        np.random.shuffle(suffle_index)
        print len(train_blur_imgs)
        #print suffle_index
        
        train_blur_imgs = []
        train_classification_mask =[]
        for i in range(0,len(suffle_index),1):
            train_blur_imgs.append(prev_train_blur_imgs[suffle_index[i]] )
            train_classification_mask.append(prev_train_classification_mask[suffle_index[i]] )

        for idx in range(0, len(train_blur_imgs) , new_batch_size):
            step_time = time.time()

            augmentation_list = [0,1]
            augmentation= random.choice(augmentation_list)
            if(augmentation ==0):
                images_and_score = tl.prepro.threading_data([_ for _ in zip(train_blur_imgs[idx: idx + new_batch_size],train_classification_mask[idx: idx + new_batch_size])],fn=crop_sub_img_and_classification_fn)                
            elif (augmentation==1):
                images_and_score = tl.prepro.threading_data([_ for _ in zip(train_blur_imgs[idx: idx + new_batch_size],train_classification_mask[idx: idx + new_batch_size])],fn=crop_sub_img_and_classification_fn_aug)
                
            #print images_and_score.shape
            imlist, clist= images_and_score.transpose((1,0,2,3,4))
            #print clist.shape
            clist = clist[:, :, :, 0]
            #print clist.shape
            clist = np.expand_dims(clist, axis=3)

            #print imlist.shape, clist.shape
            err,l1,l2,l3,l4, _ ,outmap= sess.run([loss,loss1,loss2,loss3,loss4, train_op,out], {patches_blurred: imlist, classification_map: clist})

            outmap1 = np.squeeze(outmap[1,:,:,0])
            outmap2 = np.squeeze(outmap[1, :, :, 1])
            outmap3 = np.squeeze(outmap[1, :, :, 2])

            if(idx%100 ==0):
                scipy.misc.imsave(save_dir_sample + '/input_mask.png', np.squeeze(clist[1, :, :, 0]))
                scipy.misc.imsave(save_dir_sample + '/input.png', np.squeeze(imlist[1,:,:,:]))
                scipy.misc.imsave(save_dir_sample + '/im.png', outmap1)
                scipy.misc.imsave(save_dir_sample + '/im1.png', outmap2)
                scipy.misc.imsave(save_dir_sample + '/im2.png', outmap3)

            print("Epoch [%2d/%2d] %4d time: %4.4fs, err: %.6f, loss1: %.6f,loss2: %.6f,loss3: %.6f,loss4: %.6f" % (epoch, n_epoch, n_iter, time.time() - step_time, err,l1,l2,l3,l4))
            total_loss += err
            n_iter += 1
            global_step += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, total_err: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_loss/n_iter)
        print(log)

        ## save model
        if epoch % 10== 0:
            tl.files.save_ckpt(sess=sess, mode_name='SA_net_{}.ckpt'.format(tl.global_flag['mode']), save_dir = checkpoint_dir, var_list = a_vars, global_step = global_step, printable = False)


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

    parser.add_argument('--mode', type=str, default='PG_CUHK', help='model name')
    parser.add_argument('--is_train', type=str , default='false', help='whether train or not')
    parser.add_argument('--index', type=int, default='0', help='index range 50')


    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    tl.global_flag['is_train'] = t_or_f(args.is_train)

    if tl.global_flag['is_train']:
        train_with_CUHK()
        # train_with_synthetic() # train with the CUHK dataset frist and then finetune with the synthetic dataset
    else:       
        blurmap_3classes(args.index) #pg test
       
