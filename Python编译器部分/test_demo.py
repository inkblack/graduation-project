#/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
from scipy.misc import imread, imresize
from vgg16 import VGG16
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

class model_flag():

    def __init__(self):
        self.data_dir = None
        self.im_dir = None
        self.model_dir = None
        # self.mat_dir = '/home/yql/Finetune_food172'
        # # self.train_dir = '/home/yql/Finetune_food172/checkpoint/'
        self.num_epochs = 120
        self.batch_size = 64
        self.moving_average_decay = 0.99
        self.num_epochs_per_decay = 40
        self.learning_rate_decay_factor = 0.1
        self.initial_learning_rate = 0.001
        self.batch_norm = True
        self.weights_path = None
        self.MEAN_VALUE = None
        self.skip_layers = None

FLAGS = model_flag()

numble_class = 8
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  #set the GPU, if not use the GPU set '-1'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7  #set gpu memory
mean_color=[0,0,0]
batch_size = 1
input_size = 224
x = tf.placeholder(tf.float32, [None, input_size, input_size, 3])
dropout_keep_prob = tf.placeholder(tf.float32)
keep_learning_rate =  tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)
class_name= ['枫叶', '荷花叶', '虎皮兰', '桑树叶', '松树叶', '榆树叶', '招财树', '棕树叶']
class_english_name= ['maple leaf', 'Lotus leaf', 'Tiger orchid', 'mulberry leaf', 'Loose leaf', 'Elm leaf', 'Lucky tree', 'Palm leaf']
model = VGG16(numble_class, FLAGS)
score = model.inference(x, is_training, keep_learning_rate, batch_norm=True)
output = score
softmax = tf.nn.softmax(output)
k=5
values, indices = tf.nn.top_k(softmax, k)
########修改路径########修改路径
path = '/media/isia/c1fb0319-2e81-4225-9df4-85a7db4e4c7e/isia/llh/leaf/test/zhaocaishu/'
img = imread(path+'Z.jpg', mode='RGB')
########修改路径########修改路径

img = imresize(img, [224, 224])
img_1= np.ndarray([1, 224, 224, 3])
img_1[0]=img
with tf.Session() as sess:

    # Initialize all variables
    saver = tf.train.Saver()
########修改路径########修改路径########修改路径########修改路径########修改路径########修改路径
    checkpoint = '/media/isia/c1fb0319-2e81-4225-9df4-85a7db4e4c7e/isia/llh/leaf/check_point/Leaf-vgg16-ImageNet-SGD_epoch42_0.9375.ckpt'
########修改路径########修改路径########修改路径########修改路径########修改路径########修改路径
    saver.restore(sess, checkpoint)
    [prob, ind, out] = sess.run([values, indices, output], feed_dict={x:img_1,
                                                                      is_training:False,
                                                                      keep_learning_rate:1,
                                                                          dropout_keep_prob:1.0
                                                                      })
    prob = prob[0]
    ind = ind[0]
    x_=[]
    y_=[]
    for i in range(k):
        x_.append(class_english_name[ind[i]])
        y_.append(prob[i])
        print('\tCategory Name: %s \tProbability: %.2f%%\n' %(class_name[ind[i]], prob[i]*100))
    fig = plt.figure()
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    plt.bar(x_, y_, 0.4, color='red')
    plt.xlabel('The class label')
    plt.ylabel('The accuracy')
    plt.title('The classification of the leaf')
    plt.show()
