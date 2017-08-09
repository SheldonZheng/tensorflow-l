from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir','/Users/zhenghang/Project/tensorflow-l/mnist_log/mlp','Summaries directory')
flags.DEFINE_integer('max_steps',1000,'Max run steps')

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

def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train or FLAGS.fake_data:
          xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
          k = FLAGS.dropout
        else:
          xs, ys = mnist.test.images, mnist.test.labels
          k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

FLAGS.fake_data = False
FLAGS.dropout = 0.75
FLAGS.learning_rate = 0.3
FLAGS.max_steps = 3000
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)
sess = tf.InteractiveSession()

in_units = 784
h1_units = 300

input_layer_name = 'input_layer'

hidden1_layer_name = 'hidden1_layer'

with tf.name_scope(input_layer_name):
    W1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev = 0.1))
    variable_summaries(W1,input_layer_name + "weight1")
    b1 = tf.Variable(tf.zeros([h1_units]))
    variable_summaries(b1,input_layer_name + "biais1")

with tf.name_scope(hidden1_layer_name):
    W2 = tf.Variable(tf.zeros([h1_units,10]))
    variable_summaries(W2,hidden1_layer_name + "weight2")
    b2 = tf.Variable(tf.zeros(10))
    variable_summaries(b2,hidden1_layer_name + "biais2")

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,[None,in_units])
    image_shaped_input = tf.reshape(x,[-1,28,28,1])
    tf.summary.image('input',image_shaped_input,10)
    keep_prob = tf.placeholder(tf.float32)

hidden1 = tf.nn.relu(tf.matmul(x,W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1,keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop,W2) + b2)

y_ = tf.placeholder(tf.float32,[None,10])

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))

    train_step = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
 
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y,1) , tf.argmax(y_,1))
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