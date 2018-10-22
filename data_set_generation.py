import os
from PIL import Image
import numpy as np
import random


TESTING_RATIO = 0.1
EFFECTIVE_F_SAMPLE = 1500
EFFECTIVE_M_SAMPLE = 1500

def read_files(folder,nummax):
	res = []
	files = os.listdir(folder)
	counter = 0
	for file in files:
		img = Image.open(folder + file).convert('LA')
		data = np.array(img.getdata())[:,0]
		res.append(data)
		counter += 1
		if counter == nummax:
			break
	return res


def main():
	FM_set = read_files('F/',EFFECTIVE_F_SAMPLE)
	M_set = read_files('M/',EFFECTIVE_M_SAMPLE)
	FM_set = FM_set[:EFFECTIVE_F_SAMPLE]
	M_set = M_set[:EFFECTIVE_M_SAMPLE]
	training_total = []
	testing_total = []
	num_testing = len(FM_set)*TESTING_RATIO
	flag = np.zeros([2])
	flag[0] = 1
	for index, sample in enumerate(FM_set):
		temp = []
		temp.append(sample)
		temp.append(flag)
		if index < num_testing:
			testing_total.append(temp)
		else:
			training_total.append(temp)
	num_testing = len(FM_set) * TESTING_RATIO
	flag = np.zeros([2])
	flag[1] = 1
	for index, sample in enumerate(M_set):
		temp = []
		temp.append(sample)
		temp.append(flag)
		if index < num_testing:
			testing_total.append(temp)
		else:
			training_total.append(temp)
	random.shuffle(testing_total)
	random.shuffle(training_total)
	# sp = testing_total[0][0]
	# sp = np.reshape(sp,(250,250))
	# img = Image.fromarray(sp.astype('uint8'))
	# print(testing_total[0][1])
	# img.show()
	# pass
	xtrain = []
	ytrain = []
	xtest = []
	ytest = []
	for i in training_total:
		xtrain.append(i[0])
		ytrain.append(i[1])
	for i in testing_total:
		xtest.append(i[0])
		ytest.append(i[1])
	return xtrain,ytrain,xtest,ytest




if __name__ == '__main__':
    main()