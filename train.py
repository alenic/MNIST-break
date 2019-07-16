'''
author:  Alessandro Nicolosi
website: https://github.com/alenic

MNIST training
'''

from __future__ import print_function
import tensorflow as tf
import numpy as np
import math
import random
import cv2
import pickle
import argparse
import mnist_models
from tensorflow.examples.tutorials.mnist import input_data
import augmenter as aug
import time


def main():
  parser = argparse.ArgumentParser(description='MNIST-break')
  parser.add_argument('--model', required=True, type=str, help='the classifier model as module name (e.g. linear')
  parser.add_argument('--epochs', type=int, default=20, help='number of train epochs [default: 20]')
  parser.add_argument('--batch', type=int, default=64, help='batch number [default: 64]')
  parser.add_argument('--augmentation', action='store_true', help='Activate random data augmentation')
  parser.add_argument('--seed', type=int, default=1234, help='seed of pseudorandom generator [default: 1234]')
  args = parser.parse_args()

  # =============================== Dataset =================================
  mnist = input_data.read_data_sets("./data", one_hot=True)

  n_train_samples, _ = mnist.train.images.shape
  n_test_samples, _ = mnist.test.images.shape

  image_width = 28
  image_height = 28
  n_classes = 10

  X_train = mnist.train.images.reshape((-1, image_height, image_width, 1))
  y_train = mnist.train.labels

  X_test = mnist.test.images.reshape((-1, image_height, image_width, 1))
  y_test = mnist.test.labels

  iter_epoch_train = math.ceil(n_train_samples / args.batch)
  iter_epoch_test = math.ceil(n_test_samples / args.batch)

  print('Number of epochs: ', args.epochs)
  print('Batch size: ', args.batch)
  print('Number of iteration per epoch: ', iter_epoch_train)

  if args.augmentation:
    augmenter = aug.Augmenter([(aug.RandomRotate(), 0.5),
                                (aug.RandomShift(), 0.5),
                                (aug.RandomErode(), 0.25),
                                (aug.RandomDilate(), 0.25)])

  # =============================== Model Graph =================================
  tf.reset_default_graph()
  # placeholders
  x = tf.placeholder(tf.float32, [None, image_height, image_width, 1])
  y = tf.placeholder(tf.float32, [None, n_classes])
  training_phase = tf.placeholder(tf.bool)
  
  # Neural network model
  if args.model == 'linear':
    model = mnist_models.LinearModel(n_classes).buildModel(x)
  elif args.model == 'simple_cnn':
    model = mnist_models.SimpleCNNModel(n_classes, training_phase, dropout_rate=0.4).buildModel(x)
  elif args.model == 'simple_resnet':
    model = mnist_models.SimpleResnetModel(n_classes, training_phase, dropout_rate=0.4).buildModel(x)
  else:
    raise ValueError('Invalid model')

  logits = model['logits']
  
  # =============================== Evaluation nodes ===============================
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
  correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
  correct_predictions_sum = tf.reduce_sum(tf.cast(correct_predictions, tf.float32)) 
  accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

  # =============================== Optimization Nodes ===============================
  global_step = tf.Variable(0, trainable=False)
  n_iter = args.epochs * iter_epoch_train
  
  boundaries = [int(n_iter*0.7), int(n_iter*0.9)]
  learning_rate_values = [1e-3, 2e-4, 5e-5]
  learning_rate = tf.train.piecewise_constant(global_step, boundaries, learning_rate_values)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

  # =============================== Summary nodes ==================================
  train_acc_var = tf.placeholder(tf.float32)
  train_acc_summary = tf.summary.scalar('train_accuracy', train_acc_var)
  test_acc_var = tf.placeholder(tf.float32)
  test_acc_summary = tf.summary.scalar('test_accuracy', test_acc_var)
  lr_summary = tf.summary.scalar('learning_rate', learning_rate)

  merged_summary = tf.summary.merge_all()

  # Summary writer initialization
  if args.augmentation:
    summary_folder = 'summaries_aug/'+args.model
  else:
    summary_folder = 'summaries_no_aug/'+args.model

  # =============================== Training ========================================
  init_op = tf.global_variables_initializer()
  
  with tf.Session() as sess:
    sess.run(init_op)
    
    writer = tf.summary.FileWriter(summary_folder, sess.graph)
    
    # Training loop
    for epoch in range(args.epochs):
      tic_time = time.time()
      correct_predictions_train = 0.0
      for iteration in range(iter_epoch_train):
        i1 = iteration*args.batch
        i2 = min((iteration+1)*args.batch,  n_train_samples)
        
        X_train_batch = X_train[i1:i2]
        y_train_batch = y_train[i1:i2]

        if args.augmentation:
          X_train_batch = augmenter.augment(X_train_batch)

        sess.run(train_op, feed_dict={x: X_train_batch, y: y_train_batch, training_phase: True})
        correct_predictions_train += sess.run(correct_predictions_sum, feed_dict={x: X_train_batch, y: y_train_batch, training_phase: False})

      # Compute training accuracy
      print("Epoch %d of %d" % (epoch+1, args.epochs))

      train_accuracy = correct_predictions_train/n_train_samples
      print("Training accuracy: %.6f" % (train_accuracy))

      # Compute test accuracy
      correct_predictions_test = 0.0
      for iteration in range(iter_epoch_test):
        i1 = iteration*args.batch
        i2 = min((iteration+1)*args.batch,  n_test_samples)
        
        X_test_batch = X_test[i1:i2]
        y_test_batch = y_test[i1:i2]

        correct_predictions_test += sess.run(correct_predictions_sum, feed_dict={x: X_test_batch, y: y_test_batch, training_phase: False})

      test_accuracy = correct_predictions_test/n_test_samples
      print("Test accuracy: %.6f" % (test_accuracy))
      print("Epoch computation time: ", time.time() - tic_time)

      merged_summary_out = sess.run(merged_summary, feed_dict={train_acc_var: train_accuracy, test_acc_var: test_accuracy})
      writer.add_summary(merged_summary_out, epoch)

    writer.close()
    print("Optimization Finished!")


if __name__ == '__main__':
  main()