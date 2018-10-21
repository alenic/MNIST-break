class ModelLoader(object):
  def getModel(self, model):
    if model == 'linear':
      from .linear import LinearModel
      return LinearModel()
    elif model == 'simple_cnn':
      from .simple_cnn import SimpleCNNModel
      return SimpleCNNModel()
    else:
      ValueError('Invalid model')