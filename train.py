from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import cv2
import pickle
import argparse
from models import *
from tensorflow.examples.tutorials.mnist import input_data
import augmenter as aug


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='MNIST-break')
  parser.add_argument('--model', required=True, type=str, help='the classifier model as module name (e.g. linear')
  parser.add_argument('--epochs', type=int, default=20, help='number of train epochs [default: 20]')
  parser.add_argument('--batch', type=int, default=64, help='batch number [default: 64]')
  parser.add_argument('--augment', type=int, default=1, help='1 if for every epoch is performed random data augmentation, 0 otherwise')
  parser.add_argument('--seed', type=int, default=1234, help='seed of pseudorandom generator [default: 1234]')
  args = parser.parse_args()

  # -------------------- Dataset loading -------------------------------------------
  mnist = input_data.read_data_sets("./data", one_hot=True)

  number_of_sample, number_of_feature = mnist.train.images.shape
  image_width = 28
  image_height = 28
  number_of_class = 10

  augmenter = aug.Augmenter([(aug.RandomRotate(), 1.0),
                              (aug.RandomShift(), 1.0),
                              (aug.RandomErode(), 0.25),
                              (aug.RandomDilate(), 0.25)])

  model = ModelLoader().getModel(args.model)

  X_train = mnist.train.images
  y_train= mnist.train.labels

  # -------------------- Model graph loading ---------------------------------------
  # placeholders
  x = tf.placeholder(tf.float32, [None, number_of_feature])
  y = tf.placeholder(tf.float32, [None, number_of_class])
  training = tf.placeholder(tf.bool, name='training')

  # logits
  logits = model.getLogits(x, training)

  # evaluation nodes
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
  correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
  correct_predictions_sum = tf.reduce_sum(tf.cast(correct_predictions, tf.float32)) 

  # Optimizer
  learning_rate = 1e-4
  momentum = 0.95
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

  # -------------------- Training -------------------------------------------
  init = tf.global_variables_initializer()
  n_batch_in_epoch = int(np.ceil(float(number_of_sample) / float(args.batch)))

  print('Number of epoch: ', args.epochs)
  print('Batch size: ', args.batch)
  print('Number of batch in a epoch: ', n_batch_in_epoch)

  with tf.Session() as sess:
    sess.run(init)

    # Summary writer initialization
    if args.augment:
      summary_folder = 'summaries_aug/'+args.model
    else:
      summary_folder = 'summaries_no_aug/'+args.model

    writer = tf.summary.FileWriter(summary_folder, sess.graph)
    test_acc_var = tf.placeholder(tf.float32)
    test_acc_summary = tf.summary.scalar('test_acc', test_acc_var)
    
    # Training loop
    for epoch in range(args.epochs):
      train_accuracy_sum = 0.0
      for i_batch in range(n_batch_in_epoch):
        if(i_batch == n_batch_in_epoch-1):
            end_index = X_train.shape[0]
        else:
            end_index = (i_batch+1)*args.batch

        start_index = end_index-args.batch

        if args.augment:
          batch = X_train[start_index:end_index].reshape((-1,image_height,image_width))
          aug_batch = augmenter.augment(batch)
          X_batch = aug_batch.reshape((-1,image_height*image_width))
        else:
          X_batch = X_train[start_index:end_index]

        y_Batch = y_train[start_index:end_index]

        sess.run(optimizer, feed_dict={x: X_batch, y: y_Batch, training: True})
        train_accuracy_out = sess.run(accuracy, feed_dict={x: X_batch, y: y_Batch, training: False})

        train_accuracy_sum += train_accuracy_out

      # Compute training mean accuracy
      print("Epoch %d of %d: Training mean accuracy: %.6f" % (epoch+1, args.epochs, train_accuracy_sum/n_batch_in_epoch))

      # Compute test accuracy
      i_batch = 1
      correct_predictions_sum_out = 0.0
      while i_batch * args.batch < mnist.test.images.shape[0]:
          X_test_batch, y_test_batch = mnist.test.next_batch(args.batch)
          correct_predictions_sum_out += sess.run(correct_predictions_sum, feed_dict={x: X_test_batch, y: y_test_batch, training: False})
          i_batch += 1

      # Write to summary
      test_accuracy = correct_predictions_sum_out /  mnist.test.images.shape[0]
      print("Test Accuracy:", test_accuracy)
      test_acc_summary_str  = sess.run(test_acc_summary, feed_dict={test_acc_var: test_accuracy})
      writer.add_summary(test_acc_summary_str, epoch)

    writer.close()
    print("Optimization Finished!")