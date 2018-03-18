import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_DEM = 784 #等于图片的像素
OUTPUT_DEM = 10 #输出0-9

LAYER1_NODE = 500
BATCH_SIZE = 128

LEARNING_BASE_RATE = 0.8 #基础学习率
LEARNING_RATE_DECAY = 0.99 #衰减率

LR_RATE = 0.001 #正则化系数
TRAING_STEPS = 30000 #训练次数
MOVING_AVG_DECAY = 0.99 #滑动平均衰减率

#前向传播算法
def inference(input,avg_class,w1,b1,w2,b2):
    #没有提供滑动平均类
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input,w1) + b1)
        return tf.matmul(layer1,w2) + b2
    else:
        #首先使用avg_class.average函数进行计算滑动平均值
        layer1 = tf.nn.relu(tf.matmul(input,avg_class.average(w1)) + avg_class.average(b1))
        return tf.matmul(layer1,avg_class.average(w2)) +  avg_class.average(b2)

#训练模型
def train(minist):
    x = tf.placeholder(tf.float32,[None,INPUT_DEM],name = 'x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_DEM],name = 'y-output')

    #生成影藏层
    w1 = tf.Variable(tf.truncated_normal([INPUT_DEM,LAYER1_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

    #生成输出层
    w2 = tf.Variable(tf.truncated_normal(tf.float32,[LAYER1_NODE,OUTPUT_DEM],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_DEM]))

    y = inference(x,None,w1,biases1,w2,biases2)

    #设置平滑过程中的步长
    global_step = tf.Variable(0,trianable=False)
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVG_DECAY,global_step)
    #存储参数GRAPH.Keys.trainable_variables指的是所有没有指定trainable=False的变量
    variable_average_op = variable_average.apply(tf.trainable_variables())

    #在使用了平滑之后的前向传播
    average_y = inference(x,variable_average,w1,biases1,w2,biases2)

    #计算交叉熵
    #tf.arg_max(y_,1)返回数组最大下标
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y,tf.arg_max(y_,1))

    #计算L2正则化损失函数
    L2 = tf.contrib.layers.l2_regularizer(LR_RATE)
    regu = L2(w1) + L2(w2)
    #总损失等于交叉熵损失和正则化损失和
    loss = cross_entropy + regu

    #设置指数衰减的学习率
    learning_rates = tf.train.exponential_decay(
        LEARNING_RATE_DECAY,     #基础学习率，在这个基础上递减
        global_step,            #当前迭代的轮数
        minist.train.num_examples / BATCH_SIZE, #过完所有训练数据需要迭代次数
        LEARNING_RATE_DECAY)            #学习衰减率

    #使用梯度下降来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rates).minimize(loss,global_step=global_step)

    #训练神经网络时，需要通过反向传播来更新参数，又要跟新每个参数的平滑值，这里tf提供一个tf.control_dependencies和tf.group两种机制
    #with tf.control_dependencies([train_step,variable_average_op]):
    #    train_op = tf.no_op(name='train')
    train_op = tf.group(train_step,variable_average_op)

    correct_prediction = tf.equal(tf.arg_max(average_y,1),tf.arg_max(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        validate_feed = {
            x:minist.validation.images,
            y_:minist.validation.labels
        }
        test_feed = {x:minist.test.images,y_:minist.test.labels}

        for i in range(TRAING_STEPS):
            if i%1000 == 0:
              validate_acc = sess.run(accuracy,feed_dict=validate_feed)
              print("After %d training steps,validation accuracy using average model is %g "%(i,validate_acc))
        xs,ys = minist.train.next_batch(BATCH_SIZE)
        sess.run(train_op,feed_dict={x:xs,y_:ys})
    test_acc = sess.run(accuracy,feed_dict=test_feed)
    print("After %d training steps,test accuracy using average model is %g " % (TRAING_STEPS, test_acc))

def main(argv = None):
    minist = input_data.read_data_sets("",one_hot=True)
    train(minist)

if __name__ == '__main__':
    tf.app.run()
