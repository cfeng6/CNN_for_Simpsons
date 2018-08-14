# -*- coding: utf-8 -*-
"""
Created on Sun May 13 01:17:20 2018

@author: fengc
"""

import tensorflow as tf
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

#%% Get the images path and labels
def get_files(file_dir):
    images_path=[]
    labels= []
    
    label=0
    labels_dir_key=[]
    labels_dir_value=[]
    labels_dir=os.listdir(file_dir)
    
    for label_dir in labels_dir:
        labels_dir_key.append(label_dir)
        labels_dir_value.append(label)
        label=label+1
        
    labels_dir=dict(zip(labels_dir_key,labels_dir_value))
    print(labels_dir)
    for root,sub_folders,files in os.walk(file_dir):
        for name in files:
            images_path.append(os.path.join(root,name))
            
    for image_path in images_path:
        label_name=image_path.split('\\')[-2]
        label_value=labels_dir[label_name]
        labels.append(label_value)
    
    temp = np.array([images_path,labels]).transpose()
    np.random.shuffle(temp)
    
    images_list=list(temp[:,0])
    labels_list=list(temp[:,1])
    labels_list=[int(i) for i in labels_list]

    return images_list, labels_list
#%%
#path='C:\\Users\\fengc\\Desktop\\Spring2018\\680C_dl\\simpons_dataset'
#x_train,y_train = get_files(path)
#%%
def get_batch(image,label,image_W,image_H,batch_size,capacity):
    image=tf.cast(image,tf.string)
    label=tf.cast(label,tf.int32)
    
    input_queue=tf.train.slice_input_producer([image,label])
    label=input_queue[1]
    image_contents=tf.read_file(input_queue[0])
    image=tf.image.decode_jpeg(image_contents,channels=3)
    #extract feature ----which methods?
    image=tf.image.resize_images(image,(image_W,image_H))
    #image=tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    image=tf.image.per_image_standardization(image)
    
    image_batch, label_batch=tf.train.batch([image,label],
                                            batch_size=batch_size,
                                            num_threads=64,
                                            capacity=capacity)
    
    label_batch=tf.reshape(label_batch,[batch_size])
    image_batch=tf.cast(image_batch,tf.float32)
    
    return image_batch, label_batch
#%%
