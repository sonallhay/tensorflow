# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:33:52 2017

@author: Chen
"""
#%%
import os 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image
batch_size=100
cwd='D:\data\\' 
classes={'cats','dogs'} 


#%%
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


#%%

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
                                                    min_after_dequeue=100, #出隊時最少個數
                                                    )
    return img_batch, tf.reshape(label_batch,[batch_size])
    
img_batch, label_batch = read_and_decode('cat_and_dog_train.tfrecords',batch_size)


#%%

'''
with tf.Session()  as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop() and i<1:
            # just plot one batch size
            image, label = sess.run([img_batch, label_batch])
            for j in np.arange(5):
                print('label: %d' % label[j])
                plt.imshow(image[j,:,:,:])
                plt.show()
            i+=1
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
'''

'''
##########################################################################   
''' 
#%%
def inference(images, batch_size, n_classes):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    #conv1, shape = [kernel size, kernel size, channels, kernel numbers]
    
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights', 
                                  shape = [3,3,3, 16],
                                  dtype = tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name= scope.name)
    
    #pool1 and norm1   
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm1')
    
    #conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,16,16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
    
    
    #pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,1,1,1],
                               padding='SAME',name='pooling2')
    
    
    #local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)    
    
    #local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128,128],
                                  dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
     
        
    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[n_classes],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
    
    return softmax_linear

#%%
def losses(logits, labels):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]
        
    Returns:
        loss tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss

#%%
def trainning(loss, learning_rate):
    '''Training ops, the Op returned by this function is what must be passed to 
        'sess.run()' call to cause the model to train.
        
    Args:
        loss: loss tensor, from losses()
        
    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op

#%%
def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  with tf.variable_scope('accuracy') as scope:
      correct = tf.nn.in_top_k(logits, labels, 1)
      correct = tf.cast(correct, tf.float16)
      accuracy = tf.reduce_mean(correct)
      tf.summary.scalar(scope.name+'/accuracy', accuracy)
  return accuracy


#%%
n_classes = 2
learning_rate = 0.00001


train_logits = inference(img_batch, batch_size, n_classes)
train_loss = losses(train_logits, label_batch)
train_op = trainning(train_loss, learning_rate)
train_acc = evaluation(train_logits, label_batch)


with tf.Session() as sess:
  
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
        for step in range(15000):    
            if coord.should_stop():
                break
            #img_batch, label_batch = sess.run([img_batch, label_batch])
            _, loss, acc = sess.run([train_op,train_loss,train_acc])
            if step % 100 == 0:
                print("STEP= %d, Loss= %.2f,  train accuracy = %.2f%%" %(step, loss, acc*100.0))
    except tf.errors.OutOfRangeError:
        print("Done training")
    finally:
        coord.request_stop()
    coord.join(threads)










#%%
'''
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
'''