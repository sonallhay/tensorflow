# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 12:22:42 2017

@author: 陳正勳
"""



import os 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image
batch_size=16
cwd='D:\data\\' 
classes={'cats','dogs'} 

'''
生成TFRecord 文件
'''
writer= tf.python_io.TFRecordWriter("cat_and_dog_train.tfrecords")
for index,name in enumerate(classes):
    class_path=cwd+name+'\\'
    for img_name in os.listdir(class_path): 
        img_path=class_path+img_name

        img=Image.open(img_path)
        img= img.resize((128,128))
        img_raw=img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        })) 
        writer.write(example.SerializeToString())  

writer.close()


'''
讀取TFRecord文件By 隊列
'''

def read_and_decode(filename, batch_size): 
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128, 128, 3])  
    label = tf.cast(features['label'], tf.int32) 
    img = tf.cast(img,tf.float32)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size= batch_size,#每個批次中的樣例個數
                                                    num_threads=64, 
                                                    capacity=2000, #隊列最大容量
                                                    min_after_dequeue=1999, #出隊時最少個數
                                                    )
    return img_batch, tf.reshape(label_batch,[batch_size])
    
img_batch, label_batch = read_and_decode('cat_and_dog_train.tfrecords',batch_size)


def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name = 'W')
    
def bias(shape):
    return tf.Variable(tf.constant(0.1,shape=shape), name = 'b')
    
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')
    
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                         strides = [1, 2, 2, 1],
                        padding = 'SAME')    

with tf.name_scope('C1_Conv'):
    W1 = weight([5,5,3,16])
    b1 = bias([16])
    Conv1=conv2d(img_batch, W1)+ b1
    C1_Conv = tf.nn.relu(Conv1 )
    
with tf.name_scope('C1_Pool'):
    C1_Pool = max_pool_2x2(C1_Conv)
        
with tf.name_scope('C2_Conv'):
    W2 = weight([5,5,16,36])
    b2 = bias([36])
    Conv2=conv2d(C1_Pool, W2)+ b2
    C2_Conv = tf.nn.relu(Conv2)
    
with tf.name_scope('C2_Pool'):
    C2_Pool = max_pool_2x2(C2_Conv) 
        
with tf.name_scope('D_Flat'):
    D_Flat = tf.reshape(C2_Pool, shape=[batch_size, 36864])
        
with tf.name_scope('D_Hidden_Layer'):
    W3 = weight([36864, 128])
    b3 = bias([128])
    D_Hidden = tf.nn.relu(tf.matmul(D_Flat, W3) + b3)
    D_Hidden_Dropout = tf.nn.dropout(D_Hidden, keep_prob = 0.8)
        
with tf.name_scope('Output_Layer'):
    W4 = weight([128,2])
    b4 = bias([2])
    y_predict= tf.add(tf.matmul(D_Hidden_Dropout, W4), b4)
        
with tf.name_scope('optimizer'):
    loss_function = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                                   (logits = y_predict,
                                    labels = label_batch))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss_function)
with tf.name_scope('Evaluate_model'):
    correct_prediction = tf.nn.in_top_k(y_predict, label_batch, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
init = tf.global_variables_initializer()

from time import time
startTime=time()

with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    try:
        for step in range(15000):    
            if coord.should_stop():
                break
            if step % 50 == 0:
                _, loss, acc = sess.run([optimizer,loss_function,accuracy])
                print("STEP= %d, Loss= %.2f, Accuracy=%.2f" %(step, loss, acc))
        duration =time()-startTime
        print("Train Finished takes:",duration) 
    except tf.errors.OutOfRangeError:
        print("Done training")
    finally:
        coord.request_stop()
    coord.join(threads)
    
    


