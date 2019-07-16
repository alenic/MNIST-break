import tensorflow as tf

class SimpleCNNModel():
  def __init__(self, n_classes, training_phase, dropout_rate=0.4):
    self.n_classes = n_classes
    self.training_phase = training_phase
    self.dropout_rate = dropout_rate

  def buildModel(self, x):
    nodes = {}
    nodes['conv1'] = tf.layers.conv2d(x, filters=32, kernel_size=[5, 5], padding="same", activation=None)
    nodes['pool1'] = tf.layers.max_pooling2d(inputs=nodes['conv1'], pool_size=[2, 2], strides=2)
    nodes['conv2'] = tf.layers.conv2d(nodes['pool1'], filters=64, kernel_size=[3, 3], padding="same", activation=None)
    nodes['pool2'] = tf.layers.max_pooling2d(inputs=nodes['conv2'], pool_size=[2, 2], strides=2)
    nodes['pool2_flat'] = tf.contrib.layers.flatten(nodes['pool2'])
    nodes['dense'] = tf.layers.dense(inputs=nodes['pool2_flat'], units=1024, activation=tf.nn.relu)
    nodes['dropout'] = tf.layers.dropout(inputs=nodes['dense'], rate=self.dropout_rate, training=self.training_phase)
    nodes['logits'] = tf.layers.dense(inputs=nodes['dropout'], units=self.n_classes)

    return nodes