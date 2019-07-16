import tensorflow as tf

class LinearModel():
  def __init__(self, n_classes):
    self.n_classes = n_classes

  def buildModel(self, x):
    nodes = {}
    nodes['flatten'] = tf.contrib.layers.flatten(x)
    nodes['logits'] = tf.layers.dense(nodes['flatten'], self.n_classes)

    return nodes