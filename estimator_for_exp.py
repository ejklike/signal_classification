import argparse
from datetime import datetime
import time
import sys
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score, recall_score, confusion_matrix

import dataloader
import func

FLAGS = None

# INFO, DEBUG, ERROR, FATAL
tf.logging.set_verbosity(tf.logging.FATAL)

def save_the_evaluation(ev, fname, iter_no):
  fname_without_dt, dt = fname.split('__')
  result = '{},{},{},{:.2f},{:.2f},{:.2f}'.format(
      fname_without_dt, dt, iter_no,
      ev['AvgFscore'] * 100, ev['NormFscore'] * 100, ev['Accuracy'] * 100)
  print('Saving result -', result)
  with open('experimental_results.csv', 'a') as fout:
    fout.write(result + '\n')


class _SaveWeightHook(tf.train.SessionRunHook):
  def __init__(self, fname, iter_up_to_now):
    fname_without_dt, dt = fname.split('__')

    folder_name = './tf_weights/' + fname_without_dt
    if not os.path.exists(folder_name):
      os.makedirs(folder_name)

    basestr = '{}/{}__{}__{}.csv'
    self._mask_fname = basestr.format(folder_name, dt, 'mask', iter_up_to_now)
    self._conv_fname = basestr.format(folder_name, dt, 'conv', iter_up_to_now)

  def before_run(self, run_context):
    graph = run_context.session.graph
    conv_weight = graph.get_tensor_by_name('conv0/weights:0')
    if FLAGS.wd == 0:
      return tf.train.SessionRunArgs(conv_weight)
    else:
      mask = graph.get_tensor_by_name('conv0/mask:0')
      return tf.train.SessionRunArgs([conv_weight, mask])
  
  def after_run(self, run_context, run_values):
    if FLAGS.wd == 0:
      conv_weight = run_values.results
    else:
      conv_weight, mask = run_values.results
      # save mask
      mask = np.squeeze(mask)
      np.savetxt(self._mask_fname, mask, delimiter=',')
            
    conv_weight = np.squeeze(conv_weight)
    filter_list = np.dsplit(conv_weight, conv_weight.shape[2])
    unified_filter = np.vstack(filter_list)
    np.savetxt(self._conv_fname, unified_filter, delimiter=',')


def main(unused_argv):
  # Load datasets
  # Get signals and labels
  if FLAGS.data_type == 'all':
    trn_data_list = [dataloader.load(which_data='s{}'.format(i), 
                                     gesture=FLAGS.gesture,
                                     for_merge=True,
                                     train=True) for i in [1, 2, 3]]
    tst_data_list = [dataloader.load(which_data='s{}'.format(i), 
                                     gesture=FLAGS.gesture,
                                     for_merge=True,
                                     train=False) for i in [1, 2, 3]]
    
    signals_trn = np.concatenate([data[0] for data in trn_data_list], axis=0)
    labels_trn = np.concatenate([data[1] for data in trn_data_list], axis=0)
    signals_tst = np.concatenate([data[0] for data in tst_data_list], axis=0)
    labels_tst = np.concatenate([data[1] for data in tst_data_list], axis=0)
 
  else:
    signals_trn, labels_trn = dataloader.load(which_data=FLAGS.data_type, 
                                              gesture=FLAGS.gesture, 
                                              train=True)
    signals_tst, labels_tst = dataloader.load(which_data=FLAGS.data_type, 
                                              gesture=FLAGS.gesture, 
                                              train=False)

  # Set model params
  model_params = dict(
    learning_rate=FLAGS.learning_rate,
    # regularization type and strength
    wd=FLAGS.wd,
    # convolutional layer
    window_size=5,
    n_conv=1,
    n_filter=100,
    # fully connected layer
    n_fully_connected=1,
    # pooling
    pooling_size=2,
    pooling_stride=1,
    # n_labels
    n_labels=labels_trn.shape[1],
  ) # model0.+n_conv1.+n_filter.+n_fc1.*/eval

  model_id = 'GEST' if FLAGS.gesture is True else 'LOCO'
  model_id += '_%s' % FLAGS.data_type.upper()
  model_id += '_wd{}'.format(FLAGS.wd)
  # model_id += '_n_conv%s' % model_params['n_conv']
  # model_id += '_n_filter%s' % model_params['n_filter']
  # model_id += '_n_fc%s' % model_params['n_fully_connected']
  
  dt_now = datetime.now().strftime('%Y%m%d_%H%M%S')
  fname = '{}__{}'.format(model_id, dt_now)
  print('-'*5, model_id, '-'*5)
  print('-'*5, dt_now, '-'*5)

  # Model dir
  model_dir = './tf_models/{}/{}'.format(model_id , dt_now)
  # if FLAGS.restart is True:
  if tf.gfile.Exists(model_dir):
    tf.gfile.DeleteRecursively(model_dir)
  tf.gfile.MakeDirs(model_dir)

  # Instantiate Estimator
  estimator = tf.estimator.Estimator(
      model_fn=func.model_fn, 
      params=model_params,
      model_dir=model_dir)

  # Input functions
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'signals': signals_trn},
      y=labels_trn,
      batch_size=FLAGS.batch_size,
      num_epochs=None,
      shuffle=True)
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'signals': signals_tst},
      y=labels_tst,
      num_epochs=1,
      shuffle=False)

  # Train and test ==> record test summary
  # We iterate train and evaluation to save summaries
  if FLAGS.train is True:
    for i in range(FLAGS.steps // FLAGS.log_freq):
      iter_up_to_now = i * FLAGS.log_freq
      print('-'*10, 'Begin training - iteration', iter_up_to_now, '-'*10)
      estimator.train(
          input_fn=train_input_fn,
          steps=FLAGS.log_freq)

      # Evaluate and save the result
      iter_up_to_now = (i + 1) * FLAGS.log_freq
      ev = estimator.evaluate(
        input_fn=test_input_fn,
        hooks=[_SaveWeightHook(fname, iter_up_to_now)])
      save_the_evaluation(ev, fname, iter_up_to_now)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # choosing data
  parser.add_argument(
      'data_type', 
      type=str, 
      default='samsung',
      help='choosing data: opp_s#/samsung')

  # gpu allocation
  parser.add_argument(
      '--gpu_no', 
      type=str, 
      default=None,
      help='gpu device number')

  parser.add_argument(
      '--wd', 
      type=float, 
      default=0,
      help='weight decaying factor')

  # learning parameters
  parser.add_argument(
      '--learning_rate', 
      type=float, 
      default=0.005,
      help='initial learning rate')
  parser.add_argument(
      '--batch_size', 
      type=int, 
      default=1500,
      help='batch size')
  parser.add_argument(
      '--steps', 
      type=int, 
      default=50000,
      help='step size')
  parser.add_argument(
      '--log_freq', 
      type=int, 
      default=5000,
      help='log frequency')
  
  parser.add_argument(
      '--train', 
      type=bool, 
      nargs='?',
      default=False, #default
      const=True, #if the arg is given
      help='train or just test')
  # parser.add_argument(
  #     '--restart', 
  #     type=bool, 
  #     nargs='?',
  #     default=False, #default
  #     const=True, #if the arg is given
  #     help='restart the training')
  parser.add_argument(
      '--gesture', 
      type=bool,
      nargs='?',
      default=False, #default
      const=True, #if the arg is given
      help='Gesture....')


  FLAGS, unparsed = parser.parse_known_args()

  if FLAGS.gpu_no is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_no
  else:
    of.environ['CUDA_VISIBLE_DEVICES'] = dict(
      all='0',
      s1='1',
      s2='2',
      s3='3',
    )[FLAGS.data_type]

  tf.app.run(main=main) # , argv=[sys.argv[0]] + unparsed