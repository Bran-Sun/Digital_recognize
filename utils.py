import numpy as np
import random

def load_data(filename):
	print("Loading train data...")
	data = []
	with open(filename) as f:
		for idx, line in enumerate(f):
			if idx == 0:
				pass
			else:
				items = line.split(',')
				item = np.array(items, np.float32)
				image = item[1:].reshape((28, 28))
				image = image / 255.0
				label = item[0]
				data.append({'image': image, 'label': label})
	train_data = []
	valid_data = []
	train_data = data[:37000]
	valid_data = data[37000:]
	return train_data, valid_data

def load_test(filename):
	print("load test data...")
	data = []
	with open(filename) as f:
		for idx, line in enumerate(f):
			if idx == 0:
				pass
			else:
				items = line.split(',')
				item = np.array(items, np.float32)
				image = item.reshape((28,28))
				image = image / 255.0
				data.append({'image': image})
	return data

def gen_train_data(data):
	image = []
	label = []
	for item in data:
		image.append(item['image'])
		label.append(item['label'])
	batch_data = {}
	batch_data['image'] = np.array(image, np.float32)
	batch_data['label'] = label
	return batch_data

def train(sess, model, data, batch_size, trainable):
	random.shuffle(data)
	st, ed = 0, 0
	train_acc, train_loss = 0.0, 0.0
	turns = 0
	while ed < len(data):
		st = ed
		if (ed + batch_size) <= len(data):
			ed += batch_size
		else:
			ed = len(data)
		batch_data = gen_train_data(data[st:ed])
		loss, acc = model.train_step(sess, batch_data, trainable)
		train_acc += acc
		train_loss += loss
		turns += 1
	return train_acc / turns, train_loss / turns

def gen_test_data(data):
	image = []
	for item in data:
		image.append(item['image'])
	data = {}
	data['image'] = image
	return data

def get_test_label(sess, model, data, batcb_size):
	st, ed = 0, 0
	label = []
	while ed < len(data):
		st = ed
		if (ed + batch_size) <= len(data):
			ed += batch_size
		else:
			ed = len(data)
		batch_data = gen_test_data(data[st:ed])
		batch_label = model.inference(sess, batch_data)
		label =label + batch_data

	with open('test_label.csv', 'w') as f:
		f.write("ImageId,Label\n")
		for i in range(len(label)):
			f.write("%d,%d\n" % (i, label[i]))
