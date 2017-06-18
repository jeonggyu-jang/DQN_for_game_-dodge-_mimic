import numpy as np
import tensorflow as tf
import math
# Reusable DQN
class DQN:

		def __init__(self, session, input_size, output_size, name="main"):
				self.session = session
				self.input_size = input_size
				self.roi_width = math.sqrt(input_size)
				self.output_size = output_size
				self.net_name = name
				self._build_network()

		def _build_network(self, h_size=16, l_rate=0.0001):
				with tf.variable_scope(self.net_name):
						self._X = tf.placeholder(tf.float32, [None,int(self.roi_width),int(self.roi_width),3], name="input_x")
						W1 = tf.Variable(tf.random_normal([5, 5, 3, 32], stddev=0.01))
						L1 = tf.nn.conv2d(self._X, W1, strides=[1, 3, 3, 1], padding='VALID')
						L1 = tf.nn.relu(L1)
						
						W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
						L2 = tf.nn.conv2d(L1, W2, strides=[1, 2, 2, 1], padding='VALID')
						L2 = tf.nn.relu(L2)
						
						L3_flat = tf.reshape(L2, [-1, 64 * 6 * 6])
						
						W4 = tf.get_variable("W4", shape=[64 * 6 * 6, 256],initializer=tf.contrib.layers.xavier_initializer())
						b4 = tf.Variable(tf.random_normal([256]))
						L4 = tf.matmul(L3_flat, W4) + b4
						
						W5 = tf.get_variable("W5", shape=[256, 9],initializer=tf.contrib.layers.xavier_initializer())
						b5 = tf.Variable(tf.random_normal([9]))
						logits = tf.matmul(L4, W5) + b5
						
						self._Qpred = logits

				self._Y = tf.placeholder(
						shape=[None, self.output_size], dtype=tf.float32)

				self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
				
				self._train = tf.train.RMSPropOptimizer(
						learning_rate=l_rate,momentum=0.95).minimize(self._loss)

				self.saver=tf.train.Saver(max_to_keep=None)
		def predict(self, state):
				x = np.reshape(state,[1,int(self.roi_width),int(self.roi_width),3])
				return self.session.run(self._Qpred, feed_dict={self._X: x})
		
		def update(self, x_stack, y_stack):
			return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack}) 
