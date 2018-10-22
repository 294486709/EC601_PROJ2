import tensorflow as tf
import data_set_generation
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
from random import shuffle
# mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)
NUM_Class = 2
NUM_Layer_1 = 1000
NUM_Layer_2 = 1000
NUM_Layer_3 = 1000
NUM_Layer_4 = 1000
NUM_Layer_5 = 1000
NUM_Layer_6 = 1000
NUM_Layer_7 = 1000
NUM_Layer_8 = 1000
NUM_Layer_9 = 1000
NUM_Layer_0 = 1000
NUM_SAMPLE_SIZE = 250*250
BATCH_SIZE = 200
NUM_EPOCH = 200

x = tf.placeholder('float', [None, NUM_SAMPLE_SIZE])
y = tf.placeholder('float', [None, NUM_Class])

def nn_model(data):
	Layer_0 = {'weights':tf.Variable(tf.random_normal([NUM_SAMPLE_SIZE, NUM_Layer_0])),
			   'biases':tf.Variable(tf.random_normal([NUM_Layer_0]))}
	Layer_1 = {'weights':tf.Variable(tf.random_normal([NUM_Layer_0, NUM_Layer_1])),
			   'biases':tf.Variable(tf.random_normal([NUM_Layer_1]))}
	Layer_2 = {'weights':tf.Variable(tf.random_normal([NUM_Layer_1, NUM_Layer_2])),
			   'biases':tf.Variable(tf.random_normal([NUM_Layer_2]))}
	Layer_3 = {'weights':tf.Variable(tf.random_normal([NUM_Layer_2, NUM_Layer_3])),
			   'biases':tf.Variable(tf.random_normal([NUM_Layer_3]))}
	Layer_4 = {'weights':tf.Variable(tf.random_normal([NUM_Layer_3, NUM_Layer_4])),
			   'biases':tf.Variable(tf.random_normal([NUM_Layer_4]))}
	Layer_5 = {'weights':tf.Variable(tf.random_normal([NUM_Layer_4, NUM_Layer_5])),
			   'biases':tf.Variable(tf.random_normal([NUM_Layer_5]))}
	Layer_6 = {'weights':tf.Variable(tf.random_normal([NUM_Layer_5, NUM_Layer_6])),
			   'biases':tf.Variable(tf.random_normal([NUM_Layer_6]))}
	Layer_7 = {'weights':tf.Variable(tf.random_normal([NUM_Layer_6, NUM_Layer_7])),
			   'biases':tf.Variable(tf.random_normal([NUM_Layer_7]))}
	Layer_8 = {'weights':tf.Variable(tf.random_normal([NUM_Layer_7, NUM_Layer_8])),
			   'biases':tf.Variable(tf.random_normal([NUM_Layer_8]))}
	Layer_9 = {'weights':tf.Variable(tf.random_normal([NUM_Layer_8, NUM_Layer_9])),
			   'biases':tf.Variable(tf.random_normal([NUM_Layer_9]))}
	Out_put = {'weights':tf.Variable(tf.random_normal([NUM_Layer_9, NUM_Class])),
			   'biases':tf.Variable(tf.random_normal([NUM_Class]))}

	l0 = tf.add(tf.matmul(data, Layer_0['weights']), Layer_0['biases'])
	l0 = tf.nn.relu(l0)

	l1 = tf.add(tf.matmul(l0, Layer_1['weights']), Layer_1['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, Layer_2['weights']), Layer_2['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, Layer_3['weights']), Layer_3['biases'])
	l3 = tf.nn.relu(l3)

	l4 = tf.add(tf.matmul(l3, Layer_4['weights']), Layer_4['biases'])
	l4 = tf.nn.relu(l4)

	l5 = tf.add(tf.matmul(l4, Layer_5['weights']), Layer_5['biases'])
	l5 = tf.nn.relu(l5)

	l6 = tf.add(tf.matmul(l5, Layer_6['weights']), Layer_6['biases'])
	l6 = tf.nn.relu(l6)

	l7 = tf.add(tf.matmul(l6, Layer_7['weights']), Layer_7['biases'])
	l7 = tf.nn.relu(l7)

	l8 = tf.add(tf.matmul(l7, Layer_8['weights']), Layer_8['biases'])
	l8 = tf.nn.relu(l8)

	l9 = tf.add(tf.matmul(l8, Layer_9['weights']), Layer_9['biases'])
	l9 = tf.nn.relu(l9)

	lout = tf.add(tf.matmul(l9, Out_put['weights']),Out_put['biases'])


	return lout

def next_batch(idx, xtrain,ytrian):
	# if idx + BATCH_SIZE > len(xtrain):
	# 	idx = 0
	com = list(zip(xtrain,ytrian))
	shuffle(com)
	xtrain[:],ytrian[:] = zip(*com)
	resx = xtrain[:idx]
	resy = ytrian[:idx]
	return resx, resy

def init_variables(session):
	session.run(tf.initialize_all_variables())


def train_nn(x,xtrain,ytrain,xtest,ytest, y):
	prediction = nn_model(x)
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
		save_path = saver.save(sess,"nn/result.ckpt")
		print('saved!')



def main():
	xtrain, ytrain,xtest,ytest = data_set_generation.main()
	train_nn(x,xtrain,ytrain,xtest,ytest, y)

if __name__ == '__main__':
	main()




