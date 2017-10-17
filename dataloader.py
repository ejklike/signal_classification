import pickle

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# # batch generation
# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000
# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def load(which_data, train=True, gesture=True, for_merge=False):
  if train:
    data_fname = './data/{}_train.p'.format(which_data)
  else:
    data_fname = './data/{}_test.p'.format(which_data)

  raw_data = pickle.load(open(data_fname, 'rb'))
  signals = raw_data['signals']

  if gesture is True:
    labels = raw_data['labels_gesture']
  else:
    labels = raw_data['labels_locomotion']

  print('this is {} {} {} data!'.format(
                              'Gesture' if gesture else 'Locomotion', 
                              which_data,
                              'TRN' if train is True else 'TST'), 
        signals.shape, 
        labels.shape)

  # for merged data
  if for_merge is True and which_data != 's1':
    print('delete some sensor...')
    signals = np.concatenate([signals[:, :, :21], signals[:, :, 24:]], axis=2)

  signals = signals.astype(np.float32)
  labels = labels.astype(np.float32)


  return signals, labels

    # gesture label order
    #      0, 504605, 504608, 504611, 504616, 504617, 504619, 504620,
    # 505606, 506605, 506608, 506611, 506616, 506617, 506619, 506620,
    # 507621, 508612
    # locomotion label order
    # 0, 101, 102, 104, 105

    # LABEL DESC.
    # Unique index   -   Track name   -   Label name
    # 101   -   Locomotion   -   Stand
    # 102   -   Locomotion   -   Walk
    # 104   -   Locomotion   -   Sit
    # 105   -   Locomotion   -   Lie
    # 506616   -   Gestures   -   Open_Door1
    # 506617	 -   Gestures   -   Open_Door2
    # 504616   -   Gestures   -   Close_Door1
    # 504617   -   Gestures   -   Close_Door2
    # 506620   -   Gestures   -   Open_Fridge
    # 504620   -   Gestures   -   Close_Fridge
    # 506605	 -   Gestures   -   Open_Dishwasher
    # 504605	 -   Gestures   -   Close_Dishwasher
    # 506619   -   Gestures   -   Open_Drawer1
    # 504619   -   Gestures   -   Close_Drawer1
    # 506611   -   Gestures   -   Open_Drawer2
    # 504611   -   Gestures   -   Close_Drawer2
    # 506608   -   Gestures   -   Open_Drawer3
    # 504608   -   Gestures   -   Close_Drawer3
    # 508612   -   Gestures   -   Clean_Table
    # 507621   -   Gestures   -   Drink_Cup
    # 505606   -   Gestures   -   Toggle_Switch

    # def binary_to_dummies(arr):
    #     arr = arr.reshape(-1, 1)
    #     return np.concatenate((1-arr, arr), axis=1)
    # labels = binary_to_dummies(labels_one_col)
    # class_count = np.sum(labels, axis=0)
    # print('class imbalance: {}, {} ({:.03}%)'.format(
    #     *class_count, 
    #     class_count[1] / np.sum(class_count) * 100)
    # )

  # elif which_data == 'samsung':
  #   raw_data = pickle.load(open(data_fname, 'rb'))
  #   signals = raw_data['signals']
  #   labels = raw_data['labels']
