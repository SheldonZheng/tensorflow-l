from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir','/Users/zhenghang/Project/tensorflow-l/mnist_log/cnn','Summaries directory')
FLAGS.fake_data = False
FLAGS.dropout = 0.5
FLAGS.learning_rate = 1e-4
FLAGS.max_steps = 20000

def variable_summaries(var,name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name,mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name,stddev)
        tf.summary.scalar('max/' + name,tf.reduce_max(var))
        tf.summary.scalar('min/' + name,tf.reduce_min(var))
        tf.summary.histogram(name,var)

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train or FLAGS.fake_data:
          xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
          k = FLAGS.dropout
        else:
          xs, ys = mnist.test.images, mnist.test.labels
          k = 1.0
        return {x: xs, y_: ys, keep_prob: k}



sess = tf.InteractiveSession()

with tf.name_scope('input') :
    x = tf.placeholder(tf.float32,[None,784], name = 'x_input')
    x_image = tf.reshape(x,[-1,28,28,1])
    tf.summary.image('input',x_image,10)
    # true y
    y_ = tf.placeholder(tf.float32,[None,10],name = 'y_input')
    keep_prob = tf.placeholder(tf.float32)

conv1_ln = 'conv1'
with tf.name_scope(conv1_ln):
    W_conv1 = weight_variable([5,5,1,32])
    variable_summaries(W_conv1,conv1_ln + '/weights')
    b_conv1 = bias_variable([32])
    variable_summaries(b_conv1,conv1_ln + '/biases')
    preactivate_conv1 = conv2d(x_image,W_conv1) + b_conv1
    tf.summary.histogram(conv1_ln + '/pre_activations',preactivate_conv1)
    h_conv1 = tf.nn.relu(preactivate_conv1)
    tf.summary.histogram(conv1_ln + '/activations-relu',h_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    tf.summary.histogram(conv1_ln + '/after-pool',h_pool1)

conv2_ln = 'conv2'
with tf.name_scope(conv2_ln):
    W_conv2 = weight_variable([5,5,32,64])
    variable_summaries(W_conv2,conv2_ln + '/weights')
    b_conv2 = bias_variable([64])
    variable_summaries(b_conv2,conv2_ln + '/biases')
    preactivate_conv2 = conv2d(h_pool1,W_conv2) + b_conv2
    tf.summary.histogram(conv2_ln + '/pre_activations',preactivate_conv2)
    h_conv2 = tf.nn.relu(preactivate_conv2)
    tf.summary.histogram(conv2_ln + '/activations-relu',h_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    tf.summary.histogram(conv2_ln + '/after-pool',h_pool2)

   


fc1_ln = 'fc1'
with tf.name_scope(fc1_ln):
    W_fc1 = weight_variable([7 * 7 * 64,1024])
    variable_summaries(W_fc1,fc1_ln + '/weights')
    b_fc1 = bias_variable([1024])
    variable_summaries(b_fc1,fc1_ln + '/biases')
    h_pool2_flat = tf.reshape(h_pool2,[-1, 7*7*64])
    variable_summaries(h_pool2_flat,fc1_ln + '/h_pool2_flat')
    preactivate_fc1 = tf.matmul(h_pool2_flat,W_fc1) + b_fc1
    tf.summary.histogram(fc1_ln + '/pre_activations',preactivate_fc1)
    h_fc1 = tf.nn.relu(preactivate_fc1)
    tf.summary.histogram(fc1_ln + '/activations',h_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

fc2_ln = 'fc2'
with tf.name_scope(fc2_ln):
    W_fc2 = weight_variable([1024, 10])
    variable_summaries(W_fc2,fc2_ln + '/weights')
    b_fc2 = bias_variable([10])
    variable_summaries(b_fc2,fc2_ln + '/biases')
    temp = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
    y_conv = tf.nn.softmax(temp)
    
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),reduction_indices=[1]))
    
train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)


with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y_conv,1) , tf.argmax(y_,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy',accuracy)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',sess.graph)
test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
tf.global_variables_initializer().run()

for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
    else :
            summary, _ = sess.run([merged, train_step],feed_dict=feed_dict(True))
            train_writer.add_summary(summary,i)