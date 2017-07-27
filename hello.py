from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


# print(mnist.train.images.shape,mnist.train.labels.shape)
# print(mnist.test.images.shape,mnist.test.labels.shape)
# print(mnist.validation.images.shape,mnist.validation.labels.shape)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir','/Users/zhenghang/Project/tensorflow-l/mnist_log','Summaries directory')

def train() :
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

    sess = tf.InteractiveSession()

    with tf.name_scope('input') :
        x = tf.placeholder(tf.float32,[None,784], name = 'x_input')
        image_shaped_input = tf.reshape(x,[-1,28,28,1])
        tf.summary.image('input',image_shaped_input,10)
        # true y
        y_ = tf.placeholder(tf.float32,[None,10],name = 'y_input')


    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

# predict y  hypothesis function
    y = tf.nn.softmax(tf.matmul(x,W) + b)

    with tf.name_scope('cross_entropy'):
#cost function
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y,1) , tf.argmax(y_,1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy',accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',sess.graph)
    tf.global_variables_initializer().run()
    tf.initialize_all_variables().run()

    for i in range(1000) :
        batch_xs,batch_ys = mnist.train.next_batch(100)
        summary,_ = sess.run([merged,train_step],feed_dict={x : batch_xs, y_ : batch_ys})
        train_writer.add_summary(summary,i)
        #train_step.run({x : batch_xs, y_ : batch_ys})
        #print(accuracy.eval({x : mnist.test.images, y_ : mnist.test.labels}))

if __name__ == '__main__':
    train()

# def variable_summaries(var,name):
#     with tf.name_scope('summaries'):
#         mean = tf.reduce_mean(var)
#         tf.scalar_summary('mean/' + name,mean)
#         with tf.name_scope('stddev'):
#             stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean))
#
#         tf.scalar_summary('stddev/' + name,stddev)
#         tf.scalar_summary('max/' + name,tf.reduce_max(var))
#         tf.scalar_summary('min/' + name,tf.reduce_min(var))
#         tf.histogram_summart(name,var)
