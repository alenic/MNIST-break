import tensorflow as tf


class SimpleCNNModel():
  def __init__(self):
    self.dropout_rate = 0.75
    self.img_width = 28
    self.img_height = 28
    self.img_channel = 1

  def getLogits(self, x, training):
    img = tf.reshape(x, [-1, self.img_height, self.img_width, self.img_channel])
    conv1 = tf.layers.conv2d(img, filters=32, kernel_size=[5, 5], padding="same", activation=None)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=[3, 3], padding="same", activation=None)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.contrib.layers.flatten(pool2)
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=training)
    logits = tf.layers.dense(inputs=dropout, units=10)

    return logits