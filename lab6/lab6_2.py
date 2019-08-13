# softmax_cross_entropy_with_logits
# 동물의 종류 예측하기
import numpy as np
import tensorflow as tf

xy = np.loadtxt('./data-04-zoo.csv',delimiter=',',dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

nb_classes = 7

X = tf.placeholder(tf.float32, shape =[None,16])
Y = tf.placeholder(tf.int32, shape =[None,1])

# 파일에 출력값이 0~6으로 표현 되어 있기 때문에 onehot으로 바꾸어 주어야한다.
# tf.one_hot은 n차원이 input이면 n+1로 output한다.
# reshape를 사용하여 차원을 맞춰준다. -1은 none을 의미
Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([16,nb_classes]),name = "weight")
b = tf.Variable(tf.random_normal([nb_classes]),name = "bias")

logits = tf.matmul(X,W)+b
hypothesis = tf.nn.softmax(logits)

# cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y_one_hot)

cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

# 예측한 값 확인 하기 위해서
prediction = tf.argmax(hypothesis,1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict = {X:x_data, Y:y_data})
        if step % 100 ==0:
            loss, acc =sess.run([cost, accuracy], feed_dict ={X:x_data, Y:y_data})
            print("Step : {:5}\t Loss: {:.3f}\tAcc:{:.2%}".format(step, loss, acc))
    
    pred = sess.run(prediction, feed_dict = {X:x_data})
    for p,y in zip(pred, y_data.flatten()):
        print("[{}]Prediction : {} True Y : {}".format(p== int(y),p, int(y)))

