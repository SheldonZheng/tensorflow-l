import cifar10,cifar10_input
import tensorflow as tf
import numpy as np
import time
import math

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir','/Users/zhenghang/Project/tensorflow-l/cifar-log/cnn','Summaries directory')
FLAGS.fake_data = False
FLAGS.dropout = 0.5
FLAGS.learning_rate = 1e-4
FLAGS.max_steps = 3000
FLAGS.batch_size = 128

max_steps = 3000
batch_size = 128
#data_dir = '/Users/zhenghang/Project/tensorflow-l/cifar-10-batches-bin'
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'

cifar10.maybe_download_and_extract()

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

def variable_with_weight_loss(shape,stddev,wl):
    var = tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var),wl,name='weight_loss')
        tf.add_to_collection('loss',weight_loss)
    return var

def loss(logits,labels):
    labels = tf.cast(labels,tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,labels = labels,name = 'cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(cross_entropy_mean, tf.float32))
    tf.summary.scalar('accuracy',accuracy)
    tf.add_to_collection('losses',cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'),name='total_loss')

images_train,labels_train = cifar10_input.distorted_inputs(data_dir = data_dir ,batch_size = batch_size)

images_test,labels_test = cifar10_input.inputs(eval_data = True, data_dir = data_dir ,batch_size = batch_size)

with tf.name_scope('input'):
    image_holder = tf.placeholder(tf.float32,[batch_size,24,24,3])
    tf.summary.image('input_image',image_holder)
    label_holder = tf.placeholder(tf.int32,[batch_size])

conv1_ln = 'conv1'
with tf.name_scope(conv1_ln):
    weight1 = variable_with_weight_loss(shape = [5,5,3,64],stddev = 5e-2,wl = 0.0)
    variable_summaries(weight1,conv1_ln + '/weight')
    kernel1 = tf.nn.conv2d(image_holder,weight1 ,[1,1,1,1],padding='SAME')
    variable_summaries(kernel1,conv1_ln + '/kernel')
    bias1 = tf.Variable(tf.constant(0.0,shape=[64]))
    variable_summaries(bias1,conv1_ln + '/bias')
    conv1 = tf.nn.relu(tf.nn.bias_add(kernel1,bias1))
    variable_summaries(conv1,conv1_ln + '/conv')
    pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides = [1,2,2,1],padding = 'SAME')
    variable_summaries(pool1,conv1_ln + '/pool')

    norm1 = tf.nn.lrn(pool1,4,bias = 1.0,alpha = 0.001 / 9.0,beta = 0.75)
    variable_summaries(norm1,conv1_ln + '/norm')

conv2_ln = 'conv2'
with tf.name_scope(conv2_ln):
    weight2 = variable_with_weight_loss(shape=[5,5,64,64], stddev = 5e-2,wl = 0.0)
    variable_summaries(weight2,conv2_ln + '/weight')
    kernel2 = tf.nn.conv2d(norm1,weight2,[1,1,1,1],padding='SAME')
    variable_summaries(kernel2,conv2_ln + '/kernel')
    bias2 = tf.Variable(tf.constant(0.1,shape=[64]))
    variable_summaries(bias2,conv2_ln + '/bias')
    conv2 = tf.nn.relu(tf.nn.bias_add(kernel2,bias2))
    variable_summaries(conv2,conv2_ln + '/conv')
    norm2 = tf.nn.lrn(conv2,4,bias = 1.0 ,alpha = 0.001 / 9.0,beta = 0.75)
    variable_summaries(norm2,conv2_ln + '/norm')
    pool2 = tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,2,2,1],padding = 'SAME')
    variable_summaries(pool2,conv2_ln + '/pool')

fc1_ln = 'fc1'
with tf.name_scope(fc1_ln):
    reshape = tf.reshape(pool2,[batch_size,-1])
    #tf.summary.scalar('fc1_reshape',reshape)
    dim = reshape.get_shape()[1].value
    weight3 = variable_with_weight_loss(shape=[dim,384],stddev=0.04,wl=0.004)
    variable_summaries(weight3,fc1_ln + '/weight')
    bias3 = tf.Variable(tf.constant(0.1,shape=[384]))
    variable_summaries(bias3,fc1_ln + '/bias')
    local3 = tf.nn.relu(tf.matmul(reshape,weight3) + bias3)
    variable_summaries(local3,fc1_ln + '/local')

fc2_ln = 'fc2'
with tf.name_scope(fc2_ln):
    weight4 = variable_with_weight_loss(shape=[384,192],stddev=0.04,wl=0.004)
    variable_summaries(weight4, fc2_ln + '/weight')
    bias4 = tf.Variable(tf.constant(0.1,shape=[192]))
    variable_summaries(bias4, fc2_ln + '/bias')
    local4 = tf.nn.relu(tf.matmul(local3,weight4) + bias4)
    variable_summaries(local4,fc2_ln + '/local')

out_ln = 'out'
with tf.name_scope(out_ln):
    weight5 = variable_with_weight_loss(shape=[192,10],stddev = 1/192.0,wl=0.0)
    variable_summaries(weight5, out_ln + '/weight')
    bias5 = tf.Variable(tf.constant(0.0,shape=[10]))
    variable_summaries(bias5,out_ln + '/bias')
    logits = tf.add(tf.matmul(local4,weight5),bias5)
    variable_summaries(logits,out_ln + '/logits')

loss = loss(logits,label_holder)

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

top_k_op = tf.nn.in_top_k(logits,label_holder,1)

sess = tf.InteractiveSession()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',sess.graph)
test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
tf.global_variables_initializer().run()

tf.train.start_queue_runners()

for step in range(max_steps):
    start_time = time.time()
    image_batch,label_batch = sess.run([images_train,labels_train])
    _,loss_value = sess.run([train_op,loss],feed_dict = {image_holder :image_batch,label_holder:label_batch})
    duration = time.time() - start_time
    if step % 10 == 0:
        summary = sess.run(merged,feed_dict = {image_holder :image_batch,label_holder:label_batch})
        train_writer.add_summary(summary,step)
        summary_test,predictions = sess.run([merged,top_k_op],feed_dict = {image_holder:image_batch,label_holder:label_batch})
        test_writer.add_summary(summary_test,step)
        example_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        format_str = ('step %d ,loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step,loss_value,example_per_sec,sec_per_batch))


num_examples = 10000
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0

while step < num_iter:
    image_batch,label_batch = sess.run([images_test,labels_test])
    summary_test,predictions = sess.run([merged,top_k_op],feed_dict = {image_holder:image_batch,label_holder:label_batch})
    test_writer.add_summary(summary,step)
    true_count += np.sum(predictions)
    step += 1

precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)
