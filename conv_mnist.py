""" Train a convolutional neural network to categorise MNIST digits
"""

import sys
import argparse

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weights')


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='biases')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    with tf.name_scope('Input'):
        x = tf.placeholder(tf.float32, [None, 784])
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_images')
        y_ = tf.placeholder(tf.float32, [None, 10], name='labels')
        tf.summary.image('x_images', x_image)

    with tf.name_scope('Conv_1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('Conv_2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('FC_1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('Dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('FC_2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

    with tf.name_scope('Predictions'):
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    with tf.name_scope('Cross_Entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('Training'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('Accuracy'):
        correct = tf.equal(tf.arg_max(y_, 1), tf.arg_max(y_conv, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.summary_dir, graph=sess.graph)

    for step in range(20000):
        batch = mnist.train.next_batch(50)
        train_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5}
        if step % 100 == 0:
            summary = sess.run(merged_summary, feed_dict=train_dict)
            writer.add_summary(summary, step)
        sess.run(train_step, feed_dict=train_dict)

    test_dic = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}
    print("Test accuracy: %g" % sess.run(accuracy, feed_dict=test_dic))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data',
                        help='Directory for caching MNIST data')
    parser.add_argument('--summary_dir', default='./log',
                        help='Directory for storing summary data to view in Tensorboard')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
