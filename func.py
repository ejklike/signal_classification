import os

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.ops import nn
from tensorflow.python.summary import summary

import metrics_wrapper

padding = 'VALID'
dtype = tf.float32

def _add_hidden_layer_summary(value, tag):
  summary.scalar('%s/fraction_of_zero_values' % tag, nn.zero_fraction(value))
  summary.histogram('%s/activation' % tag, value)

# def variable_summaries(var):
#   """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
#   with tf.name_scope('summaries'):
#     mean = tf.reduce_mean(var)
#     summary.scalar('mean', mean)
#     with tf.name_scope('stddev'):
#       stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#     summary.scalar('stddev', stddev)
#     summary.scalar('max', tf.reduce_max(var))
#     summary.scalar('min', tf.reduce_min(var))
#     summary.histogram('histogram', var)


def _variable_on_cpu(name, 
                     shape, 
                     initializer=xavier_initializer(dtype=dtype)):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _conv_layer(x, shape, wd, name=None):
  """conv layer """
  name = 'conv' if name is None else name
  with tf.variable_scope(name) as scope:
    if wd > 0:
      # [window_size, sensor_size, 1, n_filter]
      kernel = _variable_on_cpu('before_mask_weights', shape=shape)
      # [1, sensor_size, 1, 1]
      mask = _variable_on_cpu('mask',
                              shape=[1] + [shape[1]] + [1, 1],
                              # shape=[1, shape[1], 1, shape[3]],
                              initializer=tf.constant_initializer(0.1))
      kernel = tf.multiply(kernel, mask, name='weights') # broadcasting

      regularizer = tf.contrib.layers.l1_regularizer(wd)
      weight_regularization = tf.contrib.layers.apply_regularization(
        regularizer, weights_list=[mask])
      tf.add_to_collection('losses', weight_regularization)
    elif wd == 0:
      kernel = _variable_on_cpu('weights', shape=shape)
    else:
      raise ValueError('Given wd is {}. wd must be 0 or positive!'.format(wd))
    _add_hidden_layer_summary(kernel, scope)

    # print(kernel.name, kernel.get_shape().as_list())
    conv = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], padding=padding)
    biases = _variable_on_cpu('biases', [shape[-1]], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    activation = tf.nn.relu(pre_activation, name=scope.name)

  return activation


def _pool_layer(x, pooling_size, pooling_stride, name=None):
  name = 'pool' if name is None else name
  with tf.name_scope(name) as name_scope:
    x = tf.nn.max_pool(x, ksize=[1, pooling_size, 1, 1], 
                        strides=[1, pooling_stride, 1, 1],
                        padding=padding, name='pool1')
  return x


def _flatten(x, name=None):
  name = 'flatten' if name is None else name
  _, n_time, n_sensor, n_filter = x.get_shape().as_list()
  x = tf.reshape(x, [-1, n_time * n_sensor * n_filter], name='flatten')
  return x


def _fully_conn_layer(x, out_dim, activation_fn=tf.nn.relu, name=None):
  name = 'fully_connected' if name is None else name
  dim = x.get_shape()[-1]
  with tf.variable_scope(name) as scope:
    weights = _variable_on_cpu('weights', shape=[dim, out_dim])
    biases = _variable_on_cpu('biases', [out_dim], tf.constant_initializer(0.1))
    if activation_fn:
      x = activation_fn(tf.matmul(x, weights) + biases, name=scope.name)
    else:
      x = tf.add(tf.matmul(x, weights), biases, name=scope.name)
  return x


def inference(signals, params):
  window_size = params['window_size']
  n_filter = params['n_filter']
  
  n_conv = params['n_conv']
  n_fully_connected = params['n_fully_connected']

  pooling_size = params['pooling_size']
  pooling_stride = params['pooling_stride']

  n_labels = params['n_labels']

  # input reshaping
  # with tf.device('/cpu:0'):
  _, time_size, sensor_size = signals.get_shape().as_list()
  signals = tf.reshape(signals, 
                      shape=[-1, time_size, sensor_size, 1], 
                      name='reshaped_input_signal')

  # conv0 with weight regularization
  x = _conv_layer(signals, 
                  [window_size, sensor_size, 1, n_filter], 
                  wd=params['wd'], 
                  name='conv0') ####
  x = _pool_layer(x, pooling_size, pooling_stride, name='pool0')
  # print('conv0', x.get_shape().as_list())

  # conv1 and more
  for i in range(1, n_conv):
    variable_size = x.get_shape().as_list()[2]
    x = _conv_layer(x, [window_size, variable_size, n_filter, n_filter],
                    wd=None, name='conv%d'%i)
    x = _pool_layer(x, pooling_size, pooling_stride, name='pool%d'%i)
    # print('conv%d'%i, x.get_shape().as_list())

  # flatten
  x = _flatten(x, name='flatten')
  # print('flatten size', x.get_shape())
  
  # fully connected
  for i in range(n_fully_connected):
    n_hidden = x.get_shape().as_list()[1] //2
    x = _fully_conn_layer(x, n_hidden, name='fully_conn%d'%i)
  
  # final logit
  logits = _fully_conn_layer(x, n_labels, activation_fn=None, name='softmax_linear')

  # # check variables
  # for v in tf.trainable_variables():
  #   print(v.name, ':', v.get_shape())

  return logits


def model_fn(features, labels, mode, params):
  """Model function for Estimator."""

  signals = features['signals']
  logits = inference(signals, params)

  # Provide an estimator spec for `ModeKeys.PREDICT`.
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={'logits': logits})

  # Calculate the average cross entropy loss across the batch.
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
  tf.add_to_collection('losses', cross_entropy_mean)
  # print(tf.get_collection('losses'))
  total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

  optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=params['learning_rate'])
  train_op = optimizer.minimize(
      loss=total_loss, global_step=tf.train.get_global_step())

  # for evaluation
  true_y, pred_y = tf.argmax(labels, axis=1), tf.argmax(logits, axis=1)

  ac = tf.metrics.accuracy(true_y, pred_y)
  af = metrics_wrapper.mean_per_class_fscore(
        true_y, pred_y, params['n_labels'], weight=False)
  nf = metrics_wrapper.mean_per_class_fscore(
          true_y, pred_y, params['n_labels'], weight=True)
  
  # Calculate root mean squared error as additional eval metric
  eval_metric_ops = {
      'Accuracy': ac,
      'AvgFscore': af,
      'NormFscore': nf,
  }

  # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=total_loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)


# def save_result(ev, FLAGS):
#     fname = './results.csv'
#     if not os.path.exists(fname):
#       with open(fname, 'w') as fout:
#         fout.write('data,model,wd,step,loss,accuracy,precision,recall,averagef1,')
#         fout.write('\n')

#     with open(fname, 'a') as fout:
#         fout.write('{},'.format(FLAGS.data_type))
#         fout.write('{},'.format(FLAGS.model))
#         fout.write('{},'.format(FLAGS.wd))
#         fout.write('{},'.format(tf.train.get_global_step()))
#         fout.write('{},'.format(ev['loss']))
#         fout.write('{},'.format(ev['accuracy']))
#         fout.write('{},'.format(ev['precision']))
#         fout.write('{},'.format(ev['recall']))
#         fout.write('{},'.format(ev['averagef1']))
#         fout.write('\n')

def save_the_evaluation(labels, logits, fname, iter_no):
  true = np.argmax(labels, axis=1)
  pred = np.argmax(logits, axis=1)

  fname_without_dt, dt = fname.split('__')
  result = '{},{},{},{:.2f},{:.2f},{:.2f}'.format(
      fname_without_dt, dt, iter_no,
      precision_recall_fscore_support(true, pred, 
                                      average='macro')[2] * 100,
      precision_recall_fscore_support(true, pred, 
                                      average='weighted')[2] * 100,
      accuracy_score(true, pred) * 100)
  print('-'*10, 'result saved...', result)
  with open('experimental_results.csv', 'a') as fout:
    fout.write(result + '\n')