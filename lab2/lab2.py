
import tensorflow as tf

x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# hypothesis is for linear model
hypothesis = x_train*W+b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

#gradient algorithm
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

# Launch session
sess = tf.Session()

# initializes global variables in the graph
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
# 20마다 cost, W, b 출력
    if step%20 == 0:
        print(step, sess.run(cost),sess.run(W),sess.run(b))
        
