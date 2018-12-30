
"""Routine for decoding the SUN397 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pdb
import tensorflow as tf

# Global constants describing the SUN397 data set.
NUM_CLASSES = 8


def get_im_list(source_dir, file_path):

    im_list = []
    im_labels = []
    with open(file_path, 'r') as fi:
        for line in fi:
            items = line.split('>')
            im_list.append(os.path.join(source_dir,items[0]))
            im_labels.append(int(items[1]))

    return im_list, im_labels



def ori_input(source_dir, data_type, file_path, batch_size,
             MEAN_VALUE, new_size, shuffle):

    im_list, im_labels = get_im_list(source_dir, file_path)

    # mat_all =load_mat(mat_list)
    data_size = len(im_list)
    data_lb = len(im_labels)

    height, width = new_size

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([im_list, im_labels],
                                                shuffle=False,
                                                name=data_type)
    # process input image data
    im_f = tf.read_file(input_queue[0])

    label = input_queue[1]


    im = tf.image.decode_jpeg(im_f, channels=3)

    im = tf.image.resize_images(im, [224, 224])
    # print(im.shape)
    ori_h = im.shape.as_list()[0]
    ori_w = im.shape.as_list()[1]
    # convert RGB to BGR
    float_image = tf.cast(tf.reverse(im, axis=[-1]), tf.float32)

    # Subtract off the Places mean (pre-trained VGG16 model on Places365)
    mean_image = tf.convert_to_tensor(np.tile(MEAN_VALUE, [ori_h, ori_w, 1]), tf.float32)
    float_image = tf.subtract(float_image, mean_image)

    # Crop the central [height, width] of the image.
    float_image = tf.image.resize_image_with_crop_or_pad(
        float_image, height, width)

    # Subtract off the mean and divide by the variance of the pixels.
    # float_image = tf.image.per_image_standardization(float_image)

    processed_image = tf.reshape(float_image, [height, width, 3])

    min_queue_examples = int(data_size*0.02)

    if shuffle:
        batch_input, batch_label = tf.train.shuffle_batch(
            [processed_image, label], batch_size,
            num_threads=4,
            capacity=min_queue_examples + 3 * batch_size,
            enqueue_many=False,
            allow_smaller_final_batch=True,
            min_after_dequeue=min_queue_examples + batch_size,
            name=data_type)
  


    else:
        batch_input, batch_label = tf.train.batch(
            [processed_image,label], batch_size,
            num_threads=4,
            capacity=min_queue_examples + 3 * batch_size,
            enqueue_many=False,
            allow_smaller_final_batch=True,
            name=data_type)

    return batch_input,batch_label, data_size
