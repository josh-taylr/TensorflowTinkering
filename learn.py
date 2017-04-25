from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# will be using mnist's train and test data-sets
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# define the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
init = tf.initialize_all_variables()

# placeholder for correct answers
y_ = tf.placeholder(tf.float32, [None, 10])

# cost function
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# training step
optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

sess = tf.Session()
sess.run(init)

# train
for i in range(1000):
    # load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {x: batch_X, y_: batch_Y}
    sess.run(train_step, feed_dict=train_data)

    if i % 100 == 0:
        a1, c1 = sess.run([accuracy, cross_entropy], feed_dict=train_data)
        test_data = {x: mnist.test.images, y_: mnist.test.labels}
        a2, c2 = sess.run([accuracy, cross_entropy], feed_dict=test_data)
        s = "Batch {0:>3} Train: acc: {1:.2f}%, cost: {2:.4f} Test: acc: {1:.2f}%, cost: {2:.4f}"
        print(s.format(i, a1, c1, a2, c2))
