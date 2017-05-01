from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import argparse

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

parser = argparse.ArgumentParser('Train a single layer NN on MNIST images.')
parser.add_argument('--learn-rate', dest='learn_rate', type=float, default=0.003,
                    help='Step value for the gradient descent algorithm')
parser.add_argument('--summary-dir', dest='summary_dir', default='/tmp/mnist_demo',
                    help="Directory to write summary data to")
args = parser.parse_args()
print args.learn_rate

with tf.name_scope("model"):
    x = tf.placeholder(tf.float32, [None, 784], name="images")
    tf.summary.image("input", tf.reshape(x, [-1, 28, 28, 1]), 3)
    W = tf.Variable(tf.zeros([784, 10]), name='weights')
    b = tf.Variable(tf.zeros([10]), name="biases")
    act = tf.matmul(x, W) + b
    tf.summary.histogram('weights', W)
    tf.summary.histogram('biases', b)
    tf.summary.histogram('activations', act)
    y = tf.nn.softmax(act)

# placeholder for correct answers
y_ = tf.placeholder(tf.float32, [None, 10], name="labels")

with tf.name_scope("cross_entropy"):
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope("training"):
    optimizer = tf.train.GradientDescentOptimizer(args.learn_rate)
    train_step = optimizer.minimize(cross_entropy)

with tf.name_scope("accuracy"):
    is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

# visualise graph in Tensorboard
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("{0}/learn={1}".format(args.summary_dir, args.learn_rate), graph=sess.graph)

# train
for i in range(1000):
    # load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {x: batch_X, y_: batch_Y}
    sess.run(train_step, feed_dict=train_data)

    if i % 5 == 0:
        s = sess.run(merged_summary, feed_dict=train_data)
        writer.add_summary(s, i)

    if i % 100 == 0:
        a1, c1 = sess.run([accuracy, cross_entropy], feed_dict=train_data)
        test_data = {x: mnist.test.images, y_: mnist.test.labels}
        a2, c2 = sess.run([accuracy, cross_entropy], feed_dict=test_data)
        s = "Batch {0:>3} Train: acc: {1:.2f}%, cost: {2:.4f} Test: acc: {1:.2f}%, cost: {2:.4f}"
        print s.format(i, a1, c1, a2, c2)

