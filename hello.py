from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


# print(mnist.train.images.shape,mnist.train.labels.shape)
# print(mnist.test.images.shape,mnist.test.labels.shape)
# print(mnist.validation.images.shape,mnist.validation.labels.shape)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir','/Users/zhenghang/Project/tensorflow-l/mnist_log','Summaries directory')
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

  # We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def add_layer(input_tensort,input_dim,output_dim,layer_name,act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim,output_dim])
            variable_summaries(weights,layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases,layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensort,weights) + biases
            tf.summary.histogram(layer_name + '/pre_activations',preactivate)
        activations = act(preactivate)
        tf.summary.histogram(layer_name + '/activations',activations)
        return activations



def train() :
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True,fake_data = FLAGS.fake_data)

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
        image_shaped_input = tf.reshape(x,[-1,28,28,1])
        tf.summary.image('input',image_shaped_input,10)
        # true y
        y_ = tf.placeholder(tf.float32,[None,10],name = 'y_input')
        keep_prob = tf.placeholder(tf.float32)


    #W = tf.Variable(tf.zeros([784,10]))
    #b = tf.Variable(tf.zeros([10]))

# predict y  hypothesis function
    #y = tf.nn.softmax(tf.matmul(x,W) + b)
    hidden1 = add_layer(x,784,500,'layer1')
    dropped = tf.nn.dropout(hidden1,keep_prob)
    y = add_layer(dropped,500,10,'layer2',act=tf.identity)

    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        #cost function
        #cross_entropy = tf.reduce_mean(tf.reduce_sum(y_ * tf.log(y)))
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

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
    #tf.initialize_all_variables().run()

    for i in range(FLAGS.max_steps) :
        if i % 10 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else :
            summary, _ = sess.run([merged, train_step],feed_dict=feed_dict(True))
            train_writer.add_summary(summary,i)
        #train_step.run({x : batch_xs, y_ : batch_ys})
        #print(accuracy.eval({x : mnist.test.images, y_ : mnist.test.labels}))

    train_writer.close()
    test_writer.close()

if __name__ == '__main__':
    FLAGS.fake_data = False
    FLAGS.dropout = 0.5
    FLAGS.learning_rate = 0.001
    FLAGS.max_steps = 10000
    train()
