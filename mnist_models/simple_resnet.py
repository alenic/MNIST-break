import tensorflow as tf


class SimpleResnetModel():
  def __init__(self, n_classes, training_phase, dropout_rate=0.4):
    self.n_classes = n_classes
    self.training_phase = training_phase
    self.dropout_rate = dropout_rate

  def residual_bn_relu_conv2d(self, x, filters, kernel_size):
    bn1 = tf.layers.batch_normalization(x, axis=1)
    relu1 = tf.nn.relu(bn1)
    conv1 = tf.layers.conv2d(relu1, filters=filters, kernel_size=kernel_size, strides=[1, 1], padding="same", activation=None)
    bn2 = tf.layers.batch_normalization(conv1, axis=1)
    relu2 = tf.nn.relu(bn2)
    conv2 = tf.layers.conv2d(relu2, filters=filters, kernel_size=kernel_size, strides=[1, 1], padding="same", activation=None)
    
    input_channel = x.get_shape().as_list()[-1]
    output_channel = conv2.get_shape().as_list()[-1]

    if input_channel != output_channel:
      pooled_input = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
      padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2, input_channel // 2]])
    else:
      padded_input = x

    return conv2 + padded_input

  def buildModel(self, x):
    nodes = {}
    conv1 = tf.layers.conv2d(x, filters=8, kernel_size=[5,5], strides=[2, 2], padding="same", activation=None)
    res_block1 = self.residual_bn_relu_conv2d(conv1, filters=16, kernel_size=[3,3])
    res_block2 = self.residual_bn_relu_conv2d(res_block1, filters=32, kernel_size=[3,3])
    res1_flat = tf.contrib.layers.flatten(res_block2)
    dense = tf.layers.dense(inputs=res1_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=self.dropout_rate, training=self.training_phase)
    nodes['logits'] = tf.layers.dense(inputs=dropout, units=self.n_classes)
    return nodes