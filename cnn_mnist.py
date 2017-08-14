from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir','/Users/zhenghang/Project/tensorflow-l/mnist_log/cnn','Summaries directory')
flags.DEFINE_integer('max_steps',1000,'Max run steps')

sess = tf.InteractiveSession()

