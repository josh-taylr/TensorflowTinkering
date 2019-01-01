from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None
IMAGE_SIZE = 28 * 28
LAYER_1_SIZE = 300
LAYER_2_SIZE = 200
CLASS_COUNT = 10


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    images = tf.placeholder(tf.float32, [None, IMAGE_SIZE])
    labels = tf.placeholder(tf.float32, [None, CLASS_COUNT])

    with tf.name_scope('Hidden_1'):
        weight = tf.Variable(tf.truncated_normal([IMAGE_SIZE, LAYER_1_SIZE], stddev=1.0))
        bias = tf.Variable(tf.zeros([LAYER_1_SIZE]))
        layer_1 = tf.nn.relu(tf.matmul(images, weight) + bias)

    with tf.name_scope('Hidden_2'):
        weight = tf.Variable(tf.truncated_normal([LAYER_1_SIZE, LAYER_2_SIZE], stddev=1.0))
        bias = tf.Variable(tf.zeros([LAYER_2_SIZE]))
        layer_2 = tf.nn.relu(tf.matmul(layer_1, weight) + bias)

    with tf.name_scope('Linear'):
        weight = tf.Variable(tf.truncated_normal([LAYER_2_SIZE, CLASS_COUNT]))
        bias = tf.Variable(tf.zeros(CLASS_COUNT))
        logits = tf.matmul(layer_2, weight) + bias  # un-normalised probabilities i.e logits

    with tf.name_scope('Loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        tf.summary.scalar('cross_entropy', tf.reduce_mean(cross_entropy))

    with tf.name_scope('Train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('Accuracy'):
        correct = tf.equal(tf.arg_max(labels, 1), tf.arg_max(logits, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.summary_dir, graph=sess.graph)

    for step in range(FLAGS.epic_count):
        if step % FLAGS.summary_interval == 0:
            test_dict = {images: mnist.test.images, labels: mnist.test.labels}
            summary, acc = sess.run([merged_summary, accuracy], feed_dict=test_dict)
            writer.add_summary(summary, step)
            print('Accuracy at step %s: %s' % (step, acc))
        batch = mnist.train.next_batch(50)
        train_dict = {images: batch[0], labels: batch[1]}
        sess.run(train_step, feed_dict=train_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        default='./data',
        dest='data_dir',
        help='Directory for caching MNIST data'
    )
    parser.add_argument(
        '--summary',
        default='./logs/feed_forward',
        dest='summary_dir',
        help='Directory for storing summary data to view in Tensorboard'
    )
    parser.add_argument(
        '--epics',
        type=int,
        default=20000,
        dest='epic_count',
        help='The number of iterations to run'
    )
    parser.add_argument(
        '--summary-interval',
        type=int,
        default=20,
        dest='summary_interval',
        help='The number of iterations to run before writing summary data'
    )
    FLAGS, unknown = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unknown)
