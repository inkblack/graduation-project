"""
A binary to finetune pre-trained AlexNet model on MIT67
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
from datetime import datetime
import time


import tensorflow as tf
import scipy.io as sio
from build_input import *
from vgg16 import VGG16

class model_flag():

    def __init__(self):
        self.data_dir = '/home/vipl/llh/Leaf'
        self.im_dir = '/home/vipl/llh/Leaf/images/'
        self.model_dir = '/home/vipl/llh/Leaf'
        self.num_epochs = 120
        self.batch_size = 64
        self.moving_average_decay = 0.99
        self.num_epochs_per_decay = 40
        self.learning_rate_decay_factor = 0.1
        self.initial_learning_rate = 0.001
        self.batch_norm = True
        self.weights_path = self.model_dir+'/vgg16-ImageNet.npy'
        self.MEAN_VALUE = None
        self.skip_layers = None

FLAGS = model_flag()

# Global constants describing the MIT67 data set.
IMAGE_SIZE = VGG16.IMAGE_SIZE
NUM_CLASSES = 8
IMAGENET_MEAN=[0,0,0]


display_step = 5


def inputs(data_type, file_path,IMAGE_SIZE, MEAN_VALUE, shuffle):
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    images, labels, data_size = ori_input(source_dir=FLAGS.im_dir,
                                        data_type=data_type,
                                        file_path=file_path,
                                        batch_size=FLAGS.batch_size,
                                        MEAN_VALUE=IMAGENET_MEAN,
                                        new_size=(IMAGE_SIZE, IMAGE_SIZE),
                                        shuffle=shuffle)

    return images,labels, data_size


def finetune(IMAGE_SIZE, MODEL, model_name):
    tf.reset_default_graph()
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    config = tf.ConfigProto()

    train_file = '/home/vipl/llh/Leaf/train.txt'


    test_file = '/home/vipl/llh/Leaf/test.txt'

    train_images, train_labels, train_num = inputs('Train_data', train_file,
                                                   IMAGE_SIZE, FLAGS.MEAN_VALUE, shuffle=True)
    val_images, val_labels, val_num = inputs('Test_data', test_file,
                                             IMAGE_SIZE, FLAGS.MEAN_VALUE, shuffle=False)

    # TF placeholder for graph input and output
    dtype = tf.float32
    x = tf.placeholder(dtype, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
    y = tf.placeholder(tf.int32, [None])
    keep_prob = tf.placeholder(dtype)
    is_training = tf.placeholder(tf.bool)
    # keep_prob = 0.5

    global_step = tf.contrib.framework.get_or_create_global_step()

    # Construct the AlexNet model 
    model = MODEL(NUM_CLASSES, FLAGS)

    # Calculate loss.
    score = model.inference(x,is_training, keep_prob,
                            batch_norm=FLAGS.batch_norm)
    loss = model.loss(score, y)


    # updates the model parameters.
    train_op = model.train(loss, train_num, global_step)

    # Evaluation op: Accuracy of the model
    with tf.name_scope("Accuracy"):

        correct_pred = tf.equal(tf.argmax(score, 1),
                                tf.cast(y, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype))

    # Add the accuracy to the summary
    tf.summary.scalar('Accuracy', accuracy)

    if not os.path.exists(FLAGS.train_dir):
        print('traindir is',FLAGS.train_dir)
        os.makedirs(FLAGS.train_dir)
    summaryfiles_path = os.path.join(FLAGS.train_dir, 'summaryfiles')
    checkpoints_path = os.path.join(FLAGS.train_dir, 'checkpoints')
    if not os.path.isdir(checkpoints_path):
        os.makedirs(checkpoints_path)
        print("****checkpoints path is made****")

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(summaryfiles_path)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver(max_to_keep=50)

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = np.ceil(float(train_num) /
                                       FLAGS.batch_size).astype(np.int16)
    print('Actual Train images per epoch: {}'.format(train_batches_per_epoch * FLAGS.batch_size))
    val_batches_per_epoch = np.ceil(float(val_num) /
                                     FLAGS.batch_size).astype(np.int16)
    print('Actual val images per epoch: {}'.format(val_batches_per_epoch * FLAGS.batch_size))



    # Start Tensorflow session
    with tf.Session(config=config) as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

 

        # Load the pretrained weights into the non-trainable layers
        model.load_initial_weights(sess)


        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,
                                               coord=coord)

        print("{} start training...".format(datetime.now()))
        print("{} open Tensorboard at --logdir {}".format(datetime.now(),
                                                          summaryfiles_path))

        # Loop over number of epochs
        prev_losses = []
        prev_acc = []
        epoch = 0
        lr = FLAGS.initial_learning_rate
   
        while True:
            # Train the model
            epoch_sta = '*'*15 + ' Epoch number: {} ' + '*'*15
            print(epoch_sta.format(epoch+1))
            step = 1
            while step <= train_batches_per_epoch:
                        
                if step % display_step == 1:
                    start_time = time.time()
                    ds_acc = []
                    ds_loss = []
                # run the training op
                batch_im,batch_label = sess.run([train_images, train_labels])
   
                los, acc, _, lr = sess.run([loss, accuracy, train_op, model.lr],
                                           feed_dict={x: batch_im,
                                                      y: batch_label,
                                                      keep_prob: 0.5,
                                                      is_training: True})
                ds_acc.append(acc)
                ds_loss.append(los)

                if step % display_step == 0:
                    dsm_acc = np.mean(ds_acc, dtype=np.float32)
                    dsm_loss = np.mean(ds_loss, dtype=np.float32)
                    out_str = "Step number: {} | time: {:.4f} | learning_rate: {}\n" + \
                            "Loss: \033[1;31;40m{:.4f}\033[0m, Accuracy: \033[1;31;40m{:.4f}\033[0m"
                    print(out_str.format(step, time.time()-start_time, lr,
                                         float(dsm_loss), float(dsm_acc)))
                step += 1

            # Evaluate the model on entire validation set
            print("{} Strat Evaluating".format(datetime.now()))
            eval_acc = 0.0
            eval_loss = 0.0
            eval_count = 0
            vstart_time = time.time()
            for _ in range(val_batches_per_epoch):
                batch_im,batch_label = sess.run([val_images,val_labels])

                vacc, vlos = sess.run([accuracy, loss],
                                       feed_dict={x: batch_im,
                                                  y: batch_label,
                                                  keep_prob: 1.0,
                                                  is_training: False})
                eval_acc += vacc
                eval_loss += vlos
                eval_count += 1
            eval_acc /= eval_count
            eval_loss /= eval_count
            vout_str = "time: {:.4f} | Loss = \033[1;31;40m{:.4f}\033[0m | Accuracy = \033[1;31;40m{:.4f}\033[0m\n"
            print(vout_str.format(time.time()-vstart_time, eval_loss, eval_acc))

            if len(prev_acc) > 1 and eval_acc > max(prev_acc):
                # Save checkpoint of the model
                print("{} Saving checkpoint of model...".format(datetime.now()))

                checkpoint_name = os.path.join(checkpoints_path,
                                               'Leaf-{}-SGD_epoch{}_{:.4f}.ckpt'.format(model_name, epoch+1, eval_acc))
                save_path = saver.save(sess, checkpoint_name)

                print("{} Model checkpoint saved at {}".format(datetime.now(), save_path))

            prev_acc.append(eval_acc)
            if lr < 1e-7:
                break
            epoch += 1

        print("Done training -- {} epochs limit reached".format(epoch+1))
            
        # When done, ask the threads to stop.
        coord.request_stop()
            
        # Wait for threads to finish.
        coord.join(threads)
        

def main(argv=None):

    # model_list = ['vgg16-places365', 'vgg16-places205', 'vgg16-ImageNet', 'alexnet-places365', 'alexnet-places205', 'alexnet-ImageNet']
    model_list = ['vgg16-ImageNet']
    for model_name in model_list:
        print('\033[1;31;40m{}\033[0m'.format(model_name))
        if model_name == 'vgg16-places365':
            vgg16_places365_flag()
            finetune(VGG16.IMAGE_SIZE, VGG16, model_name)
        elif model_name == 'vgg16-places205':
            vgg16_places205_flag()
            finetune(VGG16.IMAGE_SIZE, VGG16, model_name)
        elif model_name == 'vgg16-ImageNet':
            vgg16_ImageNet_flag()
            finetune(VGG16.IMAGE_SIZE, VGG16, model_name)
        # elif model_name == 'alexnet-places365':
        #     alexnet_places365_flag()
        #     finetune(Alexnet.IMAGE_SIZE, Alexnet, model_name)
        # elif model_name == 'alexnet-places205':
        #     alexnet_places205_flag()
        #     finetune(Alexnet.IMAGE_SIZE, Alexnet, model_name)
        # elif model_name == 'alexnet-ImageNet':
        #     alexnet_ImageNet_flag()
        #     finetune(Alexnet.IMAGE_SIZE, Alexnet, model_name)
        else:
            raise ValueError




def vgg16_ImageNet_flag():

    FLAGS.train_dir = FLAGS.data_dir + '/finetune_vgg16/ImageNet'
    FLAGS.initial_learning_rate = 0.001
    FLAGS.num_epochs_per_decay = 40
    FLAGS.learning_rate_decay_factor = 0.1
    FLAGS.weights_path = FLAGS.model_dir + '/vgg16-ImageNet.npy'
    FLAGS.MEAN_VALUE = IMAGENET_MEAN
    # FLAGS.skip_layers = [ 'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
    #                        'conv3_1', 'conv3_2', 'conv3_3',
    #                        'conv4_1', 'conv4_2', 'conv4_3','conv5_1', 'conv5_2', 'conv5_3']
    FLAGS.skip_layers = []
    return FLAGS






if __name__ == '__main__':
    tf.app.run()
