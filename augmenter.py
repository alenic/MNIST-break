import numpy as np
import random
import scipy

class Augmenter(object):
  def __init__(self, transformationList):
    self.transformationList = transformationList
  
  def augment(self, batch):
    new_batch = np.empty(batch.shape)
    for i in range(len(batch)):
      new_batch[i] = batch[i].copy()
      for j in range(len(self.transformationList)):
        if random.random() < self.transformationList[j][1]:
          new_batch[i] = self.transformationList[j][0].transform(new_batch[i])
    return new_batch

class RandomShift(object):
  def __init__(self, shift_x=(-3,3), shift_y=(-3,3)):
    self.shift_x = shift_x
    self.shift_y = shift_y
  
  def transform(self, image):
    x = random.randint(self.shift_x[0], self.shift_x[1])
    y = random.randint(self.shift_y[0], self.shift_y[1])
    return scipy.ndimage.shift(image, [x,y])


class RandomRotate(object):
  def __init__(self,  min_angle=-8.0, max_angle=8.0):
    self.min_angle = min_angle
    self.max_angle = max_angle
  
  def transform(self, image):
    angle = random.randint(self.min_angle, self.max_angle)
    return scipy.ndimage.rotate(image, angle, reshape=False)


class RandomErode(object):
  def __init__(self):
    pass
  
  def transform(self, image):
    return scipy.ndimage.grey_erosion(image, size=(2,2))


class RandomDilate(object):
  def __init__(self):
    pass
  
  def transform(self, image):
    return scipy.ndimage.grey_dilation(image, size=(2,2))


# ==================================== Main (used for test) ===================================
if __name__ == '__main__':
  from tensorflow.examples.tutorials.mnist import input_data
  import matplotlib.pyplot as plt
  import augmenter as aug
  import math

  def showBatch(batch):
    n_batch = len(batch)
    plt.figure()
    gridN = int(np.ceil(n_batch**0.5))

    if gridN*gridN < n_batch:
        gridM = gridN+1
    else:
        gridM = gridN

    for i in range(len(batch)):
        plt.subplot(gridM, gridN,i+1)
        plt.imshow(batch[i], cmap='gray')
    
    plt.draw()

  mnist = input_data.read_data_sets("./data", one_hot=True)
  batch = mnist.train.images[0:36].reshape((-1,28,28))

  augmenter = aug.Augmenter([(aug.RandomRotate(), 1.0),
                            (aug.RandomShift(), 1.0),
                            (aug.RandomErode(), 0.25),
                            (aug.RandomDilate(), 0.25)])
  new_batch = augmenter.augment(batch)

  showBatch(batch)

  showBatch(new_batch)

  plt.show()