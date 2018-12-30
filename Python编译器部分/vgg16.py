
"""Builds the VGG16 network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np

import tensorflow as tf


def _variable_on_cpu(name, shape, para):
    """Helper to create a Variable stored on CPU memory.
    
    Args:
        name: name of the variable
        shape: list of ints
        para: parameter for initializer
    
    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        if name == 'weights':
            # initializer = tf.truncated_normal_initializer(stddev=para, dtype=dtype)
            initializer = tf.contrib.layers.xavier_initializer(seed=1)
        else:
            initializer = tf.constant_initializer(para)
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

    Returns:
    Variable Tensor
    """
    # dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
      name,
      shape,
      stddev)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


class VGG16(object):
    # Build the AlexNet model
    IMAGE_SIZE = 224    # input images size

    def __init__(self, num_classes, model_flags):

        # Parse input arguments into class variables
        self.wd = 0.00005
        self.NUM_CLASSES = num_classes
        # self.KEEP_PROB = keep_prob
        self.batch_size = model_flags.batch_size
        self.SKIP_LAYERS = model_flags.skip_layers
        self.WEIGHTS_PATH = model_flags.weights_path
        self.num_epochs_per_decay = model_flags.num_epochs_per_decay
        self.learning_rate_decay_factor = model_flags.learning_rate_decay_factor
        self.lr = tf.Variable(model_flags.initial_learning_rate, trainable=False)
        self.lr_decay_op = self.lr.assign(
            self.lr * self.learning_rate_decay_factor)
        self.moving_average_decay = model_flags.moving_average_decay

    def inference(self, X,is_training, KEEP_PROB, batch_norm, reuse=False):

        # 1st Layer: Conv_1-2 (w ReLu) -> Pool
        conv1_1 = conv(X, 3, 3, 64, 1, 1, name='conv1_1', reuse=reuse)
        conv1_2 = conv(conv1_1, 3, 3, 64, 1, 1, name='conv1_2', reuse=reuse)
        pool1 = max_pool(conv1_2, 2, 2, 2, 2, name='pool1')

        # 2nd Layer: Conv_1-2 (w ReLu) -> Pool
        conv2_1 = conv(pool1, 3, 3, 128, 1, 1, name='conv2_1', reuse=reuse)
        conv2_2 = conv(conv2_1, 3, 3, 128, 1, 1, name='conv2_2', reuse=reuse)
        pool2 = max_pool(conv2_2, 2, 2, 2, 2, name='pool2')
               
        # 3rd Layer: Conv_1-3 (w ReLu) -> Pool
        conv3_1 = conv(pool2, 3, 3, 256, 1, 1, name='conv3_1', reuse=reuse)
        conv3_2 = conv(conv3_1, 3, 3, 256, 1, 1, name='conv3_2', reuse=reuse)
        conv3_3 = conv(conv3_2, 3, 3, 256, 1, 1, name='conv3_3', reuse=reuse)
        pool3 = max_pool(conv3_3, 2, 2, 2, 2, name='pool3')

        # 4th Layer: Conv_1-3 (w ReLu) -> Pool
        conv4_1 = conv(pool3, 3, 3, 512, 1, 1, name='conv4_1', reuse=reuse)
        conv4_2 = conv(conv4_1, 3, 3, 512, 1, 1, name='conv4_2', reuse=reuse)
        conv4_3 = conv(conv4_2, 3, 3, 512, 1, 1, name='conv4_3', reuse=reuse)
    
        pool4 = max_pool(conv4_3, 2, 2, 2, 2, name='pool4')

        # 5th Layer: Conv_1-3 (w ReLu) -> Pool
        conv5_1 = conv(pool4, 3, 3, 512, 1, 1, name='conv5_1', reuse=reuse)
        conv5_2 = conv(conv5_1, 3, 3, 512, 1, 1, name='conv5_2', reuse=reuse)
        conv5_3 = conv(conv5_2, 3, 3, 512, 1, 1, name='conv5_3', reuse=reuse)
        # conv5_3 = conv5_3 + M
        # fc_7=conv5_3
        pool5 = max_pool(conv5_3, 2, 2, 2, 2, name='pool5')
        
        
        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flat_shape = int(np.prod(pool5.get_shape()[1:]))
        flattened = tf.reshape(pool5, [-1, flat_shape])
        fc6 = fc(flattened, flat_shape, 4096, is_training,
                 name='fc6', reuse=reuse, batch_norm=batch_norm)
        fc6 = dropout(fc6, KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(fc6, 4096, 4096, is_training, name='fc7',
                 reuse=reuse, batch_norm=batch_norm)
        # fc_7=fc7
        fc7 = dropout(fc7, KEEP_PROB)

        # 8th Layer: FC and return unscaled activations
        # for tf.nn.softmax_cross_entropy_with_logits
        fc8 = fc(fc7, 4096, self.NUM_CLASSES, is_training,
                 relu=False, name='fc8', reuse=reuse)

        return fc8
    
    def load_initial_weights(self, session):

        not_load_layers = ['fc8']
        if self.WEIGHTS_PATH == 'None':
            raise ValueError('Please supply the path to a pre-trained model')

        print('Loading the weights of pre-trained model')

        # load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if the layer is one of the layers that should be reinitialized
            if op_name not in not_load_layers:
                with tf.variable_scope(op_name, reuse=True):
                    data = weights_dict[op_name]
                    # Biases
                    var = tf.get_variable('biases')
                    session.run(var.assign(data['biases']))
                    # Weights
                    var = tf.get_variable('weights')
                    session.run(var.assign(data['weights']))

        print('Loading the weights is Done.')

    def loss(self, logits, labels):
        """Add L2Loss to all the trainable variables.
        
        Add summary for "Loss" and "Loss/avg".
        Args:
            logits: Logits is output of AlexNet model.
            labels: Labels from distorted_inputs or inputs(). 1-D tensor
                    of shape [batch_size]
        Returns:
            Loss tensor of type float.
        """
        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # the weight decay terms (L2 loss)
        # List of trainable variables of the layers we want to train
        train_var_list = [v for v in tf.trainable_variables()
                          if v.name.split('/')[0] not in self.SKIP_LAYERS]
        print('********Weight Loss********')
        for train_var in train_var_list:
            if 'weights' in train_var.name:
                if self.wd is not None:
                    weight_decay = tf.multiply(tf.nn.l2_loss(train_var), self.wd, name='weight_loss')
                    tf.add_to_collection('losses', weight_decay)
                    print(train_var.name)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def _add_loss_summaries(self, total_loss):
        """Add summaries for losses in AlexNet model.
        
        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.
        
        Args:
            total_loss: Total loss from loss().
        Returns:
            loss_averages_op: op for generating moving averages of losses.
        """
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])
        
        # Attach a scalar summary to all individual losses and the total loss
        # do the same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version 
            # of the loss as the original loss name.
            tf.summary.scalar(l.op.name + '(raw)', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))
    
        return loss_averages_op

    def train(self, total_loss, num_examples, global_step):
        """Train AlexNet model.
        
        Create an optimizer and apply to all trainable variables. Add moving
        average for all trainable variables.
        
        Args:
            total_loss: Total loss from loss().
            num_examples: numbers of examples per epoch for train
            global_step: Integer Variable counting the number of training steps
                         processed.
        Returns:
            train_op: op for training.
        """
        # Variables that affect learning rate.
        num_batches_per_epoch = num_examples / self.batch_size
        decay_steps = int(num_batches_per_epoch * self.num_epochs_per_decay)
        
        # Decay the learning rate exponentially based on the number of steps.
        self.lr = tf.train.exponential_decay(self.lr,
                                             global_step,
                                             decay_steps,
                                             self.learning_rate_decay_factor,
                                             staircase=True)
        tf.summary.scalar('learning_rate', self.lr)
        
        # Generate moving averages of all losses and associated summaries.
        loss_averages_op = self._add_loss_summaries(total_loss)
        
        # List of trainable variables of the layers we want to train
        train_var_list = [v for v in tf.trainable_variables()
                          if v.name.split('/')[0] not in self.SKIP_LAYERS + ['fc8']]
        train_var_fc8 = [v for v in tf.trainable_variables()
                         if v.name.split('/')[0] == 'fc8']

        # Compute gradients of all trainable variable
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.GradientDescentOptimizer(self.lr)
            opt_fc8 = tf.train.GradientDescentOptimizer(self.lr * 10)

            grads = opt.compute_gradients(total_loss, train_var_list)
            grads_fc8 = opt_fc8.compute_gradients(total_loss, train_var_fc8)
        
        # Apply gradients.
        apply_grad_op_fc8 = opt_fc8.apply_gradients(grads_fc8)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        ops = [apply_gradient_op, apply_grad_op_fc8]
        # Add histograms for trainable variables.
        for var in train_var_list + train_var_fc8:
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads + grads_fc8:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates_op = tf.group(*update_ops)
            ops.append(updates_op)

        with tf.control_dependencies(ops):
            train_op = tf.no_op(name='train')
    
        return train_op


"""
Predefine all necessary layer for the AlexNet
"""


def conv(x, kernel_height, kernel_width, num_kernels, stride_y, stride_x, name,
         reuse=False, padding='SAME'):

    # Get number of input channels
    input_channels = int(x.get_shape()[-1])
    print(x.get_shape())

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name, reuse=reuse) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = _variable_on_cpu('weights', [kernel_height, kernel_width,
                                               input_channels, num_kernels], 1e-1)
        biases = _variable_on_cpu('biases', [num_kernels], 0.0)

        # Apply convolution function
        conv = convolve(x, weights)

        # Add biases
        bias = tf.nn.bias_add(conv, biases)

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)

        return relu


def fc(x, num_in, num_out, is_training, name, reuse=False,
       relu=True, batch_norm=False):
    with tf.variable_scope(name, reuse=reuse) as scope:

        # Create tf variable for the weights and biases
        # weights = _variable_with_weight_decay('weights', [num_in, num_out], 1e-1, wd)
        weights = _variable_on_cpu('weights', [num_in, num_out], 1e-1)
        biases = _variable_on_cpu('biases', [num_out], 1.0)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if batch_norm:
            # Adds a Batch Normalization layer
            act = tf.contrib.layers.batch_norm(act, center=True, scale=True,
                                               trainable=True, is_training=is_training,
                                               reuse=reuse, scope=scope)

        if relu:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


def max_pool(x, kernel_height, kernel_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, kernel_height, kernel_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


