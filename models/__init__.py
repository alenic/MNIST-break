class ModelLoader(object):
  def getModel(self, model):
    if model == 'linear':
      from .linear import LinearModel
      return LinearModel()
    elif model == 'simple_cnn':
      from .simple_cnn import SimpleCNNModel
      return SimpleCNNModel()
    elif model == 'simple_resnet':
      from .simple_resnet import SimpleResnetModel
      return SimpleResnetModel()
    else:
      raise ValueError('Invalid model')