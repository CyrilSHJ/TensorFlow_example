import tensorflow as tf

v1 = tf.Variable(0,dtype=tf.float32)
step = tf.Variable(0,trainable=False)
ema = tf.train.ExponentialMovingAverage(0.99,step)
maintain_averages_op = ema.apply([v1]) #定义更行变量的平均操作

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    #print(sess.run(v1))
    print(sess.run([v1,ema.average(v1)]))

    #更行变量v1
    sess.run(tf.assign(step,10000))
    sess.run(tf.assign(v1,10))
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)]))

    # 更新一次v1的滑动平均值
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))