# softmax
import tensorflow as tf

x_data = [[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]

X = tf.placeholder(tf.float32, shape =[None,4])
Y = tf.placeholder(tf.float32, shape =[None,3])
# 클래스 수가 3개
nb_classes = 3 

W = tf.Variable(tf.random_normal([4,nb_classes]),name = "weight")
b = tf.Variable(tf.random_normal([nb_classes]),name = "bias")

# softmax 사용
hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis),axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict = {X:x_data, Y:y_data})
        if step % 200 ==0:
            print(step, sess.run(cost,feed_dict = {X:x_data, Y:y_data}))


    a = sess.run(hypothesis, feed_dict = {X:[[1,11,7,9]]})
    #확률로 돌려주는데 가장 큰 것을 출력
    print(a, sess.run(tf.math.argmax(a,1)))

    # 여러개의 데이터를 예측
    all = sess.run(hypothesis, feed_dict = {X:[[1,11,7,9],[1,3,4,3],[1,1,0,1]]})
    print(all, sess.run(tf.math.argmax(all,1)))

    
