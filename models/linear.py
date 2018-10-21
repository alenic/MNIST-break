import tensorflow as tf


class LinearModel():
  def __init__(self):
    pass

  def getLogits(self, x, training):
    return tf.layers.dense(x, 10)