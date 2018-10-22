import tensorflow as tf
import data_set_generation
import numpy as np
from random import shuffle
from tqdm import tqdm

NUM_Class = 2
BATCH_SIZE = 200
DATA_SIZE = 250
fc_size = 7*7*512
NUM_EPOCH = 70

x = tf.placeholder('float', [None, DATA_SIZE*DATA_SIZE])
y = tf.placeholder('float', [None, NUM_Class])

def CNN(x):
	weights = {'w_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
			   'w_conv2': tf.Variable(tf.random_normal([3,3,32,64])),
			   'w_conv3':tf.Variable(tf.random_normal([5,5,64,128])),
			   'w_conv4':tf.Variable(tf.random_normal([3,3,128,256])),
			   'w_conv5':tf.Variable(tf.random_normal([3,3,256,512])),
			   'w_fc1': tf.Variable(tf.random_normal([fc_size,1024])),
			   'w_fc2': tf.Variable(tf.random_normal([1024, 1024])),
			   'w_fc3': tf.Variable(tf.random_normal([1024, 1024])),
			   'out': tf.Variable(tf.random_normal([1024,NUM_Class]))}
	biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
			   'b_conv2': tf.Variable(tf.random_normal([64])),
			  'b_conv3': tf.Variable(tf.random_normal([128])),
			  'b_conv4': tf.Variable(tf.random_normal([256])),
			  'b_conv5': tf.Variable(tf.random_normal([512])),
			   'b_fc1': tf.Variable(tf.random_normal([1024])),
			  'b_fc2':tf.Variable(tf.random_normal([1024])),
			  'b_fc3':tf.Variable(tf.random_normal([1024])),
			   'out': tf.Variable(tf.random_normal([NUM_Class]))}
	x = tf.reshape(x, shape=[-1,DATA_SIZE,DATA_SIZE,1])
	conv1 = tf.nn.relu(tf.nn.conv2d(x,weights['w_conv1'],strides=[1,1,1,1], padding='SAME') + biases['b_conv1'])
	conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1],padding= 'SAME')

	conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w_conv2'], strides=[1,1,1,1], padding='VALID') + biases['b_conv2'])
	conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	conv3 = tf.nn.relu(tf.nn.conv2d(conv2,weights['w_conv3'],strides=[1,1,1,1], padding='SAME') + biases['b_conv3'])
	conv3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1],padding= 'SAME')

	conv4 = tf.nn.relu(tf.nn.conv2d(conv3, weights['w_conv4'], strides=[1,1,1,1], padding='VALID') + biases['b_conv4'])
	conv4 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	conv5 = tf.nn.relu(tf.nn.conv2d(conv4, weights['w_conv5'], strides=[1,1,1,1], padding='VALID') + biases['b_conv5'])
	conv5 = tf.nn.max_pool(conv5, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



	fc1 = tf.reshape(conv5, [-1,fc_size])
	fc1 = tf.nn.relu(tf.matmul(fc1, weights['w_fc1']) + biases['b_fc1'])
	fc2 = tf.reshape(fc1, [-1,fc_size])
	fc2 = tf.nn.relu(tf.matmul(fc1, weights['w_fc2']) + biases['b_fc2'])
	fc3 = tf.reshape(fc2, [-1,fc_size])
	fc3 = tf.nn.relu(tf.matmul(fc1, weights['w_fc3']) + biases['b_fc3'])
	output = tf.matmul(fc3, weights['out']) + biases['out']

	return output

def next_batch(idx, xtrain,ytrian):
	# if idx + BATCH_SIZE > len(xtrain):
	# 	idx = 0
	com = list(zip(xtrain,ytrian))
	shuffle(com)
	xtrain[:],ytrian[:] = zip(*com)
	resx = xtrain[:idx]
	resy = ytrian[:idx]
	return resx, resy


def train_nn(x,xtrain,ytrain,xtest,ytest, y):
	prediction = CNN(x)
	cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(NUM_EPOCH):
			epoch_loss = 0
			for idx in tqdm(range(int(len(xtrain)/BATCH_SIZE))):
				epoch_x, epoch_y = next_batch(BATCH_SIZE,xtrain,ytrain)
				_, c = sess.run([optimizer,cost], feed_dict={x:epoch_x,y:epoch_y})
				epoch_loss += c
				#print("Epoch:{} completed out of {}, loss{}".format(epoch, NUM_EPOCH, epoch_loss))
				correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
				accuracy = tf.reduce_mean(tf.cast(correct,'float'))
			print('Accuracy:{}, loss{}, epoch{}'.format(accuracy.eval({x:xtest,y:ytest}),epoch_loss,epoch))
		save_path = saver.save(sess,"Cnn/result.ckpt")
		print('saved!')


def main():
	xtrain, ytrain,xtest,ytest = data_set_generation.main()
	train_nn(x,xtrain,ytrain,xtest,ytest, y)

if __name__ == '__main__':
	main()
