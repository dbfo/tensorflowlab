
import tensorflow as tf

# 1
hello = tf.constant("hello, TensorFlow")
#  노드 만들고
sess = tf.Session()
#  session 만들고
print(sess.run(hello)) 
#  session 실행


# 2
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess= tf.Session()

print("sess.run(node1, node2) : ", sess.run([node1, node2]))
print("sess.run(node3)", sess.run(node3))

# 3
a= tf.placeholder(tf.float32)
b= tf.placeholder(tf.float32)
adder_node = a+b

print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]}))
