import tensorflow as tf
from numpy.random import RandomState
#定义训练数据batch大小
batch_size = 16

#定义神经网络的各层的参数 这边定义2层的隐藏层
w1 = tf.Variable(tf.random_normal([2,4],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([4,4],stddev=1,seed=1))
w3 = tf.Variable(tf.random_normal([4,1],stddev=1,seed=1))

#定义输入信息
x = tf.placeholder(tf.float32,shape=(None,2),name='x_input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name='y_input')

#定义计算过程
c1 = tf.matmul(x,w1)
c2 = tf.matmul(c1,w2)
y = tf.matmul(c2,w3)

#定义损失函数
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.AdadeltaOptimizer(0.001).minimize(cross_entropy)

#随机生成数据
data_size = 256
rd = RandomState(1)
X = rd.rand(data_size,2)
#定义x_1 + x_2 >= 1 => _y =0 or x_1 + x_2 < 1 => _y = 1
Y = [[int(x_1+x_2<1)] for (x_1,x_2) in X]

with tf.Session() as sess:
    init_parm = tf.global_variables_initializer()
    sess.run(init_parm)
    #print(sess.run(w1))
    #print(sess.run(w2))
    #print(sess.run(w3))

    #设置循环次数
    STEPS_NUM = 5000
    for i in range(STEPS_NUM):
        start = (i*batch_size) % data_size
        end = min(start+batch_size,data_size)

        #根据获取的样本对参数进行更新
        #print(X[start:end])
        #print(Y[start:end])
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})

        #每迭代1000次查看一下交叉商
        if i%1000 == 0:
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print("after %d trian steps,cross_entropy is %g"%(i,total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))
    print(sess.run(w3))

