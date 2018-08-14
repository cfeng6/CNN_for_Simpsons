# -*- coding: utf-8 -*-
"""
Created on Sun May 13 13:09:08 2018

@author: fengc
"""

import tensorflow as tf
#%%
def conv(x,filter_height,filter_width,number_filters,stride_x,stride_y,name,padding='SAME'):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:    
        input_channels = int(x.get_shape()[-1])
        weights=tf.get_variable('weights',
                                shape=[filter_height,filter_width,input_channels,number_filters],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(
                                        stddev=0.1,dtype=tf.float32))
        tf.summary.histogram('weight',weights)
        biases=tf.get_variable('biases',
                                shape=[number_filters],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0.1))
        tf.summary.histogram('bias',biases)
        conv=tf.nn.conv2d(x,weights,strides=[1,stride_x,stride_y,1],padding=padding)
        pre_activation=tf.nn.bias_add(conv,biases)
        out=tf.nn.relu(pre_activation,name=scope.name)
    return out

def max_pool(x, filter_width, filter_height, stride_x, stride_y, name,
             padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_width, filter_height, 1],
                          strides=[1, stride_x, stride_y, 1],
                          padding=padding, name=name)
    
def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)

def fully_connect(x,batch_size,number_next,name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
        reshape=tf.reshape(x,shape=[batch_size,-1])
        dim=reshape.get_shape()[1].value
        weights=tf.get_variable('weights',
                                shape=[dim,number_next],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(
                                        stddev=0.1,dtype=tf.float32))
        tf.summary.histogram('weight',weights)
        biases=tf.get_variable('biases',
                                shape=[number_next],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0.1))
        tf.summary.histogram('bias',biases)
        fc=tf.nn.relu(tf.matmul(reshape,weights)+biases,name=scope.name)
    return fc

def softmax(x,number_next,name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
        input_channels = int(x.get_shape()[-1])
        x_reshape=tf.reshape(x,[-1,input_channels])
        weights=tf.get_variable('weights',
                                shape=[input_channels,number_next],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(
                                        stddev=0.1,dtype=tf.float32))
        tf.summary.histogram('weight',weights)
        biases=tf.get_variable('biases',
                                shape=[number_next],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0.1))
        tf.summary.histogram('bias',biases)
        softmax=tf.nn.softmax(tf.matmul(x_reshape,weights)+biases,name=scope.name)
    return softmax

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)

def inference(images, batch_size, n_classes):
    conv1=conv(images,7,7,32,1,1,name='conv1',padding='SAME')
    #norm1=lrn(conv1,2,1e-04,0.75,name='norm1')
    pool1=max_pool(conv1,3,3,2,2,name='pool1',padding='SAME')
    
    conv2=conv(pool1,5,5,64,1,1,name='conv2',padding='SAME')
    #norm2=lrn(conv2,2,1e-04,0.75,name='norm2')
    pool2=max_pool(conv2,3,3,2,2,name='pool2',padding='SAME')
    
    conv3=conv(pool2,3,3,16,1,1,name='conv3',padding='SAME')
    conv4=conv(conv3,3,3,32,1,1,name='conv4',padding='SAME')
    conv5=conv(conv4,3,3,16,1,1,name='conv5',padding='SAME')
    
    pool3=max_pool(conv5,3,3,2,2,name='pool3',padding='SAME')
    fc1=fully_connect(pool3,batch_size,1024,name='fc1')
    dropout1=dropout(fc1,0.75)
    fc2=fully_connect(dropout1,batch_size,128,name='fc2')
    dropout2=dropout(fc2,0.75)
    softmax1=softmax(dropout2,n_classes,name='softmax1')
    return softmax1
#%%
def losses(logits,labels):
    with tf.variable_scope('loss',reuse=tf.AUTO_REUSE) as scope:
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='xentropy_per_example')
        loss=tf.reduce_mean(cross_entropy,name='loss')
        tf.summary.scalar(scope.name+'/loss',loss)
    return loss
#%%
def trainning(loss,learning_rate):
    with tf.variable_scope('optimizer',reuse=tf.AUTO_REUSE):
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step=tf.Variable(0,name='global_step',trainable=False)
        train_op=optimizer.minimize(loss,global_step=global_step)
    return train_op
#%%
def evaluation(logits,labels):
    with tf.variable_scope('accuracy',reuse=tf.AUTO_REUSE) as scope:
        correct=tf.nn.in_top_k(logits,labels,1)
        correct=tf.cast(correct,tf.float16)
        accuracy=tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy',accuracy)
    return accuracy
