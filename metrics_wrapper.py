from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope

def _create_local(name, shape, collections=None, validate_shape=True,
                  dtype=dtypes.float32):
  """Creates a new local variable.
  Args:
    name: The name of the new or existing variable.
    shape: Shape of the new or existing variable.
    collections: A list of collection names to which the Variable will be added.
    validate_shape: Whether to validate the shape of the variable.
    dtype: Data type of the variables.
  Returns:
    The created variable.
  """
  # Make sure local variables are added to tf.GraphKeys.LOCAL_VARIABLES
  collections = list(collections or [])
  collections += [ops.GraphKeys.LOCAL_VARIABLES]
  return variable_scope.variable(
      lambda: array_ops.zeros(shape, dtype=dtype),
      name=name,
      trainable=False,
      collections=collections,
      validate_shape=validate_shape)


def _streaming_confusion_matrix(labels, predictions, num_classes, weights=None):
  """Calculate a streaming confusion matrix.
  Calculates a confusion matrix. For estimation over a stream of data,
  the function creates an  `update_op` operation.
  Args:
    labels: A `Tensor` of ground truth labels with shape [batch size] and of
      type `int32` or `int64`. The tensor will be flattened if its rank > 1.
    predictions: A `Tensor` of prediction results for semantic labels, whose
      shape is [batch size] and type `int32` or `int64`. The tensor will be
      flattened if its rank > 1.
    num_classes: The possible number of labels the prediction task can
      have. This value must be provided, since a confusion matrix of
      dimension = [num_classes, num_classes] will be allocated.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
  Returns:
    total_cm: A `Tensor` representing the confusion matrix.
    update_op: An operation that increments the confusion matrix.
  """
  
  # Local variable to accumulate the predictions in the confusion matrix.
  cm_dtype = dtypes.int64 if weights is not None else dtypes.float64
  total_cm = _create_local(
      'total_confusion_matrix',
      shape=[num_classes, num_classes],
      dtype=cm_dtype)

  # Cast the type to int64 required by confusion_matrix_ops.
  predictions = math_ops.to_int64(predictions)
  labels = math_ops.to_int64(labels)
  num_classes = math_ops.to_int64(num_classes)

  # Flatten the input if its rank > 1.
  if predictions.get_shape().ndims > 1:
    predictions = array_ops.reshape(predictions, [-1])

  if labels.get_shape().ndims > 1:
    labels = array_ops.reshape(labels, [-1])

  if (weights is not None) and (weights.get_shape().ndims > 1):
    weights = array_ops.reshape(weights, [-1])

  # Accumulate the prediction to current confusion matrix.
  current_cm = confusion_matrix.confusion_matrix(
      labels, predictions, num_classes, weights=weights, dtype=cm_dtype)
  update_op = state_ops.assign_add(total_cm, current_cm)
  return total_cm, update_op


def mean_per_class_fscore(labels,
                          predictions,
                          num_classes,
                          weight=False,
                          metrics_collections=None,
                          updates_collections=None,
                          name=None):
  with variable_scope.variable_scope(name, 'average_fscore',
                                     (predictions, labels)):
    # Check if shape is compatible.
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    total_cm, update_op = _streaming_confusion_matrix(
        labels, predictions, num_classes)

    def compute_mean_fscore(name, weight=False):
      """Compute the mean per class accuracy via the confusion matrix."""
      per_row_sum = true_per_class = math_ops.to_float(math_ops.reduce_sum(total_cm, axis=1))
      per_col_sum = pred_per_class = math_ops.to_float(math_ops.reduce_sum(total_cm, axis=0))
      cm_diag = true_positive = math_ops.to_float(array_ops.diag_part(total_cm))

      def _safe_div_score(numerator, denominator):
        """return zero if denominator is zero"""
        return array_ops.where(math_ops.greater(denominator, 0),
                               math_ops.div(numerator, denominator),
                               array_ops.zeros_like(denominator))
      
      precision = _safe_div_score(true_positive, pred_per_class)
      recall = _safe_div_score(true_positive, true_per_class)
      
      numerator = math_ops.scalar_mul(2, math_ops.multiply(precision, recall))
      denominator = math_ops.add(precision, recall)
      fscores = _safe_div_score(numerator, denominator)
      
      if weight is False:
        return math_ops.reduce_mean(fscores, name=name)
      else:
        sum_values = math_ops.reduce_sum(math_ops.multiply(fscores, true_per_class))
        num_values = math_ops.reduce_sum(true_per_class)
        return  math_ops.div(sum_values, num_values, name=name)

    name = 'average_fscore' if weight is False else 'normalized_fscore'
    fscore_v = compute_mean_fscore(name, weight=weight)

    if metrics_collections:
      ops.add_to_collections(metrics_collections, fscore_v)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return fscore_v, update_op