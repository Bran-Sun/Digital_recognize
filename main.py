import tensorflow as tf
import numpy as np
from utils import load_data
from utils import train, load_test
from model import Model
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.app.flags.DEFINE_integer("epoch", 32, "Number of epoch")
tf.app.flags.DEFINE_integer("class_num", 10, "number of class")

tf.app.flags.DEFINE_string("train_data", "./data/train.csv", "train data")
tf.app.flags.DEFINE_string("test_data", "./data/test.csv", "test data")
tf.app.flags.DEFINE_string("train_dir", "./train", "train directory")

tf.app.flags.DEFINE_string("opt", "SGD", "opt")

tf.app.flags.DEFINE_float("learning_rate", 0.1, "learning rate")
tf.app.flags.DEFINE_float("learning_rate_decay", 0.75, "learning rate decay")

tf.app.flags.DEFINE_boolean("is_train", True, "whether is tran")

with tf.Session() as sess:
	print(FLAGS.__flags)
	if FLAGS.is_train:
		train_data, valid_data = load_data(FLAGS.train_data)

		model = Model(FLAGS)
		model.print_parameters()
		if tf.train.get_checkpoint_state(FLAGS.train_dir):
			model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
		else:
			sess.run(tf.global_variables_initializer())
		epoch = 0
		pre_loss = 1000000.0;
		while epoch < FLAGS.epoch:
			#train data
			start_time = time.time()
			train_acc, train_loss = train(sess, model, train_data, FLAGS.batch_size, trainable = True)
			epoch_time = time.time() - start_time
			lr = model.learning_rate.eval()
			print("epoch %d time: %.4f seconds, learning_rate: %.6f\n train loss: %.6f, train accuracy: %.6f" % (epoch, epoch_time, lr, train_loss, train_acc))

			valid_acc, valid_loss = train(sess, model, valid_data, FLAGS.batch_size, trainable = False)
			print("valid loss: %.6f, valid accuracy: %.6f" % (valid_loss, valid_acc))
			if valid_loss < pre_loss:
				pre_loss = valid_loss
				model.saver.save(sess, '%s/ckp' % FLAGS.train_dir, global_step = epoch)
				sess.run(model.learning_rate_decay_op)
			epoch += 1
	else:
		model = Model(FLAGS)
		model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
		test_data = load_test(FLAGS.test_data)
		get_test_label(sess, model, test_data, )