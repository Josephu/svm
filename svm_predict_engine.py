import sys
import pdb
import logging
import time
import random # shuffle
import numpy as np
from sklearn import svm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
# from sklearn import cross_validation
from pprint import pprint # print beautifully

RESULT_ROW_ID = 9
TRAIN_SIZE = 80
C_VALUE = 1

def format_y(y):
  """ libsvm only take 1, -1 as true, false for result format """
  return {
    'N': -1,
    'O': 1
  }[y]

def format_x(array):
  """ libsvm only take number, need to convert string to number """
  number_array = []
  for e in array:
    number_array.append(float(e))
  return number_array

def format_data(row):
  """ format input array to proper format """
  result = format_y(row.pop(RESULT_ROW_ID))
  elements = format_x(row)
  return result, elements

def to_array(line):
  """ convert input string to array """
  return line.split("\n")[0].split(',') 

def main():
  if len(sys.argv) < 2:
    print('Please choose input file, eg. python prepare_data.py input.csv')
    exit(0)

  shuffled_input = []

  ys = []
  xs = []

  input_file = sys.argv[1]
  input_ptr = open(input_file)
  c_value = str(C_VALUE)
  model_file = input_file.split('.')[0] + '_c' + c_value + '.model'

  logging.basicConfig(filename='predict.log',level=logging.DEBUG, format='')

  # Load content from file and present as array
  for line in input_ptr:
    shuffled_input.append(to_array(line))

  # Shuffle input data
  random.shuffle(shuffled_input)

  # build x, y arrays
  for line in shuffled_input:
    y, x = format_data(line)
    ys.append(y)
    xs.append(x)

  # convert to numpy array
  ys = np.asarray(ys)
  xs = np.asarray(xs)

  # dir(clf) # check all methods for clf

  # train model
  clf = svm.SVC(kernel='linear', C=1, probability=True)
  clf.fit(xs[:TRAIN_SIZE], ys[:TRAIN_SIZE])

  logging.info("current time %s" % time.strftime("%c") )
  logging.info("shuffled inputs:")
  logging.info(xs)
  logging.info("support vectors:")
  logging.info(clf.support_vectors_)
  logging.info("indices of support vectors:")
  logging.info(clf.support_)
  logging.info("classes:")
  logging.info(clf.classes_)
  logging.info("number of support vectors for each class:")
  logging.info(clf.n_support_)

  # predict results
  ys_predicted_normal = clf.predict(xs[TRAIN_SIZE:])
  ys_predicted = clf.decision_function(xs[TRAIN_SIZE:])
  logging.info("predicted results:")
  logging.info(ys_predicted_normal)
  logging.info("actual results:")
  logging.info(ys[TRAIN_SIZE:])
  logging.info("predicted distances:")
  logging.info(ys_predicted)

  # accuracy
  logging.info("accuracy:")
  accuracy = clf.score(xs[TRAIN_SIZE:], ys[TRAIN_SIZE:]) * 100
  logging.info(str(accuracy) + "%")

  # precision and recall
  precision = dict()
  recall = dict()
  precision, recall, _ = precision_recall_curve(ys[TRAIN_SIZE:], ys_predicted)
  average_precision = average_precision_score(ys[TRAIN_SIZE:], ys_predicted)
  logging.info("precision:")
  logging.info(precision)
  logging.info("recall:")
  logging.info(recall)
  #pdb.set_trace()

  # Plot Precision-Recall curve
  plt.clf()
  plt.plot(recall, precision, label='Precision-Recall curve')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.ylim([0.0, 1.05])
  plt.xlim([0.0, 1.0])
  plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision))
  plt.legend(loc="lower left")
  plt.show()

#   prob  = svm_problem(ys[:TRAIN_SIZE], xs[:TRAIN_SIZE])

#   param = svm_parameter('-t 0 -c ' + c_value) # linear and c as 1, 10 or 100
#   m = svm_train(prob, param)
#   svm_save_model(model_file, m)
#   p_label, p_acc, p_val = svm_predict(ys[TRAIN_SIZE:], xs[TRAIN_SIZE:], m)

#   logging.info("Current time %s" % time.strftime("%c") )
#   logging.info('input_file = ' + input_file)
#   logging.info('model_file = ' + model_file)
#   logging.info('c = ' + c_value)
#   logging.info('train size = ' + str(TRAIN_SIZE))
#   logging.info('accuracy: ' + str(p_acc))
#   logging.info('predicted results: ' + str(p_label))
#   logging.info('----')
#   #pdb.set_trace()

if __name__ == '__main__': main()