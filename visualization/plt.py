import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,Weights) + biases

    if activation_function==None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

#define inputs shape
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

#add layer1
layer1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
#add prection layer
prection = add_layer(layer1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(prection-ys),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#init parameter
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

#loop
for i in range(1001):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 20 == 0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        pre_value = sess.run(prection,feed_dict={xs:x_data})
        lines = ax.plot(x_data,pre_value,'r-',lw=4)
        plt.pause(0.2)
sess.close()