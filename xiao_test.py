#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 21:56:37 2018

@author: Michael
"""

import numpy
import tensorflow as tf
import numpy as np
import os

import xiao_cnn

checkpoint_dir = os.getcwd() + '/model'
keep_prob = tf.placeholder(tf.float32)

image_number = 9673
total_images = 9673


def inputs(filename, batch_size):
    image, label = xiao_cnn.read_file(filename)
    images, labels = tf.train.batch([image, label],
                                    batch_size=batch_size,
                                    capacity=total_images + batch_size)
    return images, labels


def test_model():
    global image_number
    correct = 0
    while image_number > 0:
        batch_x, batch_y = sess.run([images, labels])
        batch_x = np.reshape(batch_x, [xiao_cnn.batch_size, xiao_cnn.input_size])
        acc = sess.run([correct_pred], feed_dict={xiao_cnn.X: batch_x, xiao_cnn.Y: batch_y, keep_prob: 1})
        image_number = image_number - xiao_cnn.batch_size
        correct = correct + numpy.sum(acc)
        print("Predicted %d out of %d; partial accuracy %.4f" % (correct, total_images - image_number, correct / (total_images - image_number)))
    print(correct/total_images)


logits = xiao_cnn.conv_net(xiao_cnn.X, xiao_cnn.weights, xiao_cnn.biases, keep_prob)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                        labels=xiao_cnn.Y))
optimizer = tf.train.AdamOptimizer(learning_rate=xiao_cnn.learning_rate)
train_op = optimizer.minimize(loss=loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), xiao_cnn.Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()


saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    tfrecords_name = 'validation-00000-of-00001'
    images, labels = inputs(tfrecords_name, xiao_cnn.batch_size)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

    test_model()

    coord.request_stop()
    coord.join(threads)
    sess.close()

