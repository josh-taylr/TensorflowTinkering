from __future__ import print_function

""" Train to recognise MNIST digits using a simple neural network 

"""

import sys
import argparse

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

IMAGE_SIZE = 28 * 28


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    with tf.name_scope('Model'):
        x = tf.placeholder(tf.float32, [None, IMAGE_SIZE], name='input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='output')
        W = tf.Variable(tf.zeros([IMAGE_SIZE, 10]), name='weights')
        b = tf.Variable(tf.zeros([10]), name='biases')
        y = tf.matmul(x, W) + b

        tf.summary.image('x_images', tf.reshape(x, [-1, 28, 28, 1]))
        tf.summary.histogram('weights', W)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', y)

    with tf.name_scope('Cross_Entropy'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('Training'):
        train_step = tf.train.GradientDescentOptimizer(
            FLAGS.learn_rate).minimize(cross_entropy)

    with tf.name_scope('Accuracy'):
        correct_predication = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predication, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.summary_dir, graph=sess.graph)

    for step in range(20000):
        batch = mnist.train.next_batch(100)
        train_dict = {x: batch
        [0], y_: batch[1]}
        if step % 10 == 0:
            s = sess.run(merged_summary, feed_dict=train_dict)
            writer.add_summary(s, step)
        sess.run(train_step, feed_dict=train_dict)

    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a single layer NN on MNIST images.')
    parser.add_argument('--learn-rate', dest='learn_rate', type=float, default=0.03,
                        help='Step value for the gradient descent algorithm')
    parser.add_argument('--summary-dir', dest='summary_dir', default='./logs/softmax',
                        help="Directory to write summary data to")
    parser.add_argument('--data_dir', default='./data',
                        help='Directory for storing MNIST images')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
