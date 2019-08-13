# logistic classification
import tensorflow as tf

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3] ,[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

X = tf.placeholder(tf.float32, shape =[None,2])
Y = tf.placeholder(tf.float32, shape =[None,1])

# shape : feature이 2개 출력 값이 1개
W = tf.Variable(tf.random_normal([2,1]),name = "weight")
b = tf.Variable(tf.random_normal([1]),name = "bias")

# tf.div(1.,1.+tf.exp(tf.matmul(x,W)+b))로도 표현가능하다
hypothesis = tf.sigmoid(tf.matmul(X,W)+b)

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

# 0.5보다 크면 true로 type float32인 1로 cast해준다. 
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# 예측한 값과 Y의 값이 같으면 1로 다르면 0으로 cast해준뒤 총 평균을 구한다 (예측확률)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost,train], feed_dict = {X:x_data, Y:y_data})
        if step % 200 ==0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis,predicted,accuracy], feed_dict = { X:x_data, Y:y_data})
    print("\nHypothesis : ",h,"\nCorrect (Y): ",c,"\nAccurary: ", a)
