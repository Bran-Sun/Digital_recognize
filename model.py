import tensorflow as tf

class Model:
    def __init__(self, tfFLAGS):
        self.learning_rate_decay = tfFLAGS.learning_rate_decay
        self.learning_rate = tf.Variable(tfFLAGS.learning_rate, trainable=False, dtype=tf.float32)

        self._x = tf.placeholder(tf.float32, [None, 28, 28])
        self.y = tf.placeholder(tf.int32, [None])

        self.x = tf.reshape(self._x, [-1, 28, 28, 1])

        self.W1 = tf.Variable(tf.truncated_normal(shape = [5, 5, 1, 32], stddev = 0.1))
        self.b1 = tf.Variable(tf.constant(0.1, shape = [32]))
        self.conv1 = tf.nn.conv2d(self.x, self.W1, strides = [1, 1, 1, 1], padding = "SAME") + self.b1

        #self.bn1 = batch_normalization_layer(self.conv1)
        self.re1 = tf.nn.relu(self.conv1)

        self.pool1 = tf.nn.max_pool(self.re1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")

        self.conv2_W = tf.Variable(tf.truncated_normal(shape = [5, 5, 32, 64], stddev = 0.1))
        self.conv2_b = tf.Variable(tf.constant(0.1, shape = [64]))
        self.conv2 = tf.nn.conv2d(self.pool1, self.conv2_W, strides = [1, 1, 1, 1], padding = "SAME") + self.conv2_b

        #self.bn2 = batch_normalization_layer(self.conv2)

        self.re2 = tf.nn.relu(self.conv2)

        self.pool2 = tf.nn.max_pool(self.re2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")

        #self.pro1 = tf.nn.dropout(self.pool2, self.keep_prob)

        self.lin = tf.reshape(self.pool2, [-1, 7 * 7 * 64])

        logits = tf.layers.dense(self.lin, tfFLAGS.class_num)

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits))
        self.pred = tf.cast(tf.argmax(logits, 1), tf.int32)
        self.correct_pred = tf.equal(self.pred, self.y)
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * self.learning_rate_decay)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()

        if tfFLAGS.opt == "SGD":
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, global_step = self.global_step, var_list = self.params)
        else:
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)
        self.saver = tf.train.Saver(max_to_keep = 2)

    def train_step(self, sess, data, trainable):
        input_feed = {
        self._x: data['image'],
        self.y: data['label']
        }
        if trainable:
            output_feed = [self.loss, self.acc, self.train_op]
        else:
            output_feed = [self.loss, self.acc]

        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1]

    def inference(self, sess, data):
        input_feed = {
        self._x: data['image']
        }
        output_feed = [self.pred]
        outputs = sess.run(outpl input_feed)
        return outputs[0]

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

def batch_normalization_layer(inputs, isTrain=True):
    mean, var = tf.nn.moments(inputs, [0, 1, 2])
    hat = (inputs - mean) / tf.sqrt(var)

    shape = [inputs.get_shape()[-1]]
    scale = tf.Variable(tf.ones(shape))
    shift = tf.Variable(tf.zeros(shape))

    inputs = scale * hat + shift
    return inputs