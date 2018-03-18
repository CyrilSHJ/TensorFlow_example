import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


#define paramter
INPUT_NODE = 784
OUTPUT_NOE = 10
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
BATCH_SIZE = 64

def compute_accuracy(v_xs,v_ys):
    global prection
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_pre = tf.equal(tf.arg_max(y_pre,1),tf.arg_max(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pre,tf.float32)) #cast change to true or false
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result

def weight_variable(shape):
    init = tf.truncated_normal(shape=shape,stddev=0.1)
    return tf.Variable(init)

def bias_variable(shape):
    init = tf.constant(0.1,shape=shape)
    return tf.Variable(init)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pooling(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#define palceholder
xs = tf.placeholder(tf.float32,[None,INPUT_NODE])
ys = tf.placeholder(tf.float32,[None,OUTPUT_NOE])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs,[-1,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS])

#conv1 layer
w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
h_pool1 = max_pooling(h_conv1)

#conv2 layer
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2 = max_pooling(h_conv2)

#full connect layer1
w_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat1 = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat1,w_fc1)+b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob=keep_prob)

#full connect softmax
w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    if i%50==0:
        print(compute_accuracy(
            mnist.test.images,mnist.test.labels
        ))
sess.close()