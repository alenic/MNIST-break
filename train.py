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
import argparse
import mnist_models
from tensorflow.examples.tutorials.mnist import input_data
import augmenter as aug
import time
import os

def main():
  parser = argparse.ArgumentParser(description='MNIST-break')
  parser.add_argument('--model', required=True, type=str, help='the classifier model as module name (e.g. linear')
  parser.add_argument('--epochs', type=int, default=40, help='number of train epochs [default: 20]')
  parser.add_argument('--batch', type=int, default=64, help='batch number [default: 64]')
  parser.add_argument('--augmentation', action='store_true', help='Activate random data augmentation')
  parser.add_argument('--seed', type=int, default=1234, help='seed of pseudorandom generator [default: 1234]')
  parser.add_argument('--gpu', type=int, default=0, help='GPU id (use it for multi-gpu systems)')
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
                                (aug.RandomShift(), 0.5)])

  # =============================== Model Graph =================================
  tf.reset_default_graph()
  # placeholders
  x = tf.placeholder(tf.float32, [None, image_height, image_width, 1])
  label = tf.placeholder(tf.float32, [None, n_classes])
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
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))
  correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
  cp_sum = tf.reduce_sum(tf.cast(correct_predictions, tf.float32)) 
  batch_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

  # =============================== Optimization Nodes ===============================
  global_step = tf.Variable(0, trainable=False)
  n_iter = args.epochs * iter_epoch_train
  
  boundaries = [int(n_iter*0.6), int(n_iter*0.8)]
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
  batch_acc_summary = tf.summary.scalar('batch_accuracy', batch_accuracy)

  merged_summary = tf.summary.merge([train_acc_summary, test_acc_summary, lr_summary])

  # Summary writer initialization
  if args.augmentation:
    aug_folder = 'aug'
  else:
    aug_folder = 'no_aug'
  
  model_id = time.strftime('%Y-%m-%d_%H.%M.%S')

  summary_folder = os.path.join('summaries', args.model, aug_folder, model_id) 

  # =============================== Training ========================================
  init_op = tf.global_variables_initializer()
  
  gpu_opts = tf.GPUOptions(visible_device_list=str(args.gpu))
  config = tf.ConfigProto(gpu_options=gpu_opts)

  with tf.Session(config=config) as sess:
    sess.run(init_op)
    
    writer = tf.summary.FileWriter(summary_folder, sess.graph)
    
    # Training loop
    for epoch in range(args.epochs):
      tic_time = time.time()
      cp_train = 0.0

      X_train, y_train = aug.shuffle(X_train, y_train)
      for iteration in range(iter_epoch_train):
        step = sess.run(global_step)

        i1 = iteration*args.batch
        i2 = min((iteration+1)*args.batch,  n_train_samples)
        
        X_train_batch = X_train[i1:i2]
        y_train_batch = y_train[i1:i2]

        if args.augmentation:
          X_train_batch = augmenter.augment(X_train_batch)

        feed_train = {x: X_train_batch, label: y_train_batch, training_phase: False} 
        sess.run(train_op, feed_dict=feed_train)
        
        feed_no_train = {x: X_train_batch, label: y_train_batch, training_phase: False} 
        cp_sum_out = sess.run(cp_sum, feed_dict=feed_no_train)

        if iteration % 100 == 0:
          acc_summary_out = sess.run(batch_acc_summary, feed_dict=feed_no_train)
          writer.add_summary(acc_summary_out, step)

        cp_train += cp_sum_out
      # Compute training accuracy
      print("Epoch %d of %d" % (epoch+1, args.epochs))

      train_accuracy = cp_train/n_train_samples
      print("Training accuracy: %.6f" % (train_accuracy))

      # Compute test accuracy
      correct_predictions_test = 0.0
      for iteration in range(iter_epoch_test):
        i1 = iteration*args.batch
        i2 = min((iteration+1)*args.batch,  n_test_samples)
        
        X_test_batch = X_test[i1:i2]
        y_test_batch = y_test[i1:i2]

        correct_predictions_test += sess.run(cp_sum, feed_dict={x: X_test_batch, label: y_test_batch, training_phase: False})

      test_accuracy = correct_predictions_test/n_test_samples
      print("Test accuracy: %.6f" % (test_accuracy))
      print("Epoch computation time: ", time.time() - tic_time)

      merged_summary_out = sess.run(merged_summary, feed_dict={train_acc_var: train_accuracy, test_acc_var: test_accuracy})
      writer.add_summary(merged_summary_out, epoch)

    writer.close()
    print("Optimization Finished!")


if __name__ == '__main__':
  main()