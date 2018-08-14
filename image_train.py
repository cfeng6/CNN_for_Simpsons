# -*- coding: utf-8 -*-
"""
Created on Sun May 13 01:52:47 2018

@author: fengc
"""

import os
import numpy as np
import tensorflow as tf
import image_input
import image_model1
#import image_model2
from PIL import Image
import matplotlib.pyplot as plt
#%%
n_classes=19
img_H=64
img_W=64
batch_size=128
capacity=2000
max_step=200000
learning_rate=0.0001
#%%
def run_train():
    train_dir='C:\\Users\\fengc\\Desktop\\The_simpsons_train_data'
    logs_train_dir='C:\\Users\\fengc\\Desktop\\simpsons_3rd_version\\record\\May17_1'
    train,train_label=image_input.get_files(train_dir)
    
    train_batch,train_label_batch=image_input.get_batch(train,
                                                       train_label,
                                                       img_W,
                                                       img_H,
                                                       batch_size,
                                                       capacity)
    train_logits=image_model1.inference(train_batch,batch_size,n_classes)
    train_loss=image_model1.losses(train_logits,train_label_batch)
    train_op=image_model1.trainning(train_loss,learning_rate)
    train_acc=image_model1.evaluation(train_logits,train_label_batch)
    summary_op=tf.summary.merge_all()
    
    sess=tf.Session()
    train_writer=tf.summary.FileWriter(logs_train_dir, sess.graph)
    init=tf.global_variables_initializer()
    sess.run(init)
      
    saver=tf.train.Saver()
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    
    try:
        for step in range(max_step):
            if coord.should_stop():
                break
            #_,tra_loss,tra_acc=sess.run([train_op,train_loss,train_acc])
            [_, tra_loss, tra_acc] = sess.run([train_op, train_loss, train_acc])
            #tf.variable_scope.reuse_variables()
            if step % 10 ==0:
                print('Step %d, train loss = %.2f, train accuracy = %.4f' %(step,tra_loss,tra_acc))
                summary_str=sess.run(summary_op)
                train_writer.add_summary(summary_str,step)
                
            if step % 100 == 0 or (step+1)==max_step:
                checkpoint_path=os.path.join(logs_train_dir,'model.ckpt')
                saver.save(sess,checkpoint_path,global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
#%%
#run_train()
#%%
def get_pred(img_H,img_W,n_classes):
    test_dir='C:\\Users\\fengc\\Desktop\\CNN_The_Simpsons\\test_data\\Model2_the_simpsons_test'
    logs_train_dir='C:\\Users\\fengc\\Desktop\\CNN_The_Simpsons\\train_log\\model2_exp3\\'
    
    test, test_label=image_input.get_files(test_dir)
    
    n=len(test)
    ind=np.random.randint(0,n)
    img_dir=test[ind]
    image_label=test_label[ind]
    image_array=Image.open(img_dir)
    image_array=image_array.resize([img_H,img_W])
    image_array=np.array(image_array)
    
    mygraph=tf.Graph()
    with mygraph.as_default():
        image=tf.cast(image_array,tf.float32)
        image=tf.reshape(image,[1,img_H,img_W,3])
        logit=image_model1.inference(image,1,n_classes)
        #accuracy=model.evaluation(logit,test_label_batch)
        x=tf.placeholder(tf.float32, shape=[img_H,img_W,3])
        saver=tf.train.Saver()
    
    with tf.Session(graph=mygraph) as sess:
        print("Reading checkpoints...")
        ckpt=tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step=ckpt.model_checkpoint_path.split('\\')[-1].split('-')[-1]
            saver.restore(sess,ckpt.model_checkpoint_path)
            print('loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')
            
        prediction=sess.run(logit,feed_dict={x:image_array})
        
        max_index=np.argmax(prediction)
    return image_label,max_index
#%%
def test_accuracy(batch_size):
    acc=[]
    for i in range(batch_size):
        label,pred=get_pred(64,64,19)
        if label==pred:
            acc.append(1)
        else:
            acc.append(0)
    correct=acc.count(1)
    accuracy=correct/batch_size
    print('The test accuracy is %.4f' %accuracy)
    return None
#%%evaluate_accuracy(batch_size,n_classes)
test_accuracy(batch_size)
