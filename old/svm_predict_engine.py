import sys
import pdb
import logging
import time
import random # shuffle
import numpy as np
from sklearn import svm, metrics
import matplotlib.pyplot as plt
from pprint import pprint # print beautifully
# from sklearn import cross_validation
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import average_precision_score

RESULT_ROW_ID = 9
TRAIN_SIZE = 50
C_VALUE = 1

def threshold_array(start, end, dist):
  thresholds = []
  i = float(start)
  while i <= float(end+dist):
    thresholds.append(float(i))
    i += float(dist)
  return thresholds

def format_y(y):
  """ libsvm only take 1, -1 as true, false for result format """
  return {
    'N': 1,
    'O': -1
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
  elements = format_x(row[1:])
  return result, elements

def to_array(line):
  """ convert input string to array """
  return line.split("\n")[0].split(',') 

def precision_recall_curve(ys, ys_predicted):
  # precision and recall
  precision = dict()
  recall = dict()
  precision, recall, threshold = precision_recall_curve(ys[TRAIN_SIZE:], ys_predicted)
  average_precision = average_precision_score(ys[TRAIN_SIZE:], ys_predicted)
  return precision, recall, threshold

def run_svm(clf, xs, ys):
  clf.fit(xs[:TRAIN_SIZE], ys[:TRAIN_SIZE])

  logging.info("current time %s" % time.strftime("%c") )
  logging.info("shuffled inputs:")
  logging.info(xs)
  logging.info("support vectors:")
  logging.info(clf.support_vectors_)
  # logging.info("indices of support vectors:")
  # logging.info(clf.support_)
  # logging.info("classes:")
  # logging.info(clf.classes_)
  logging.info("number of support vectors for each class: " + str(clf.n_support_))
  logging.info("the importance of each vector")
  logging.info(clf.dual_coef_)
  logging.info("threshold: "+ str(clf._intercept_))

  thresholds = threshold_array(-0.3, 0.3, 0.0001)

  accuracies = []
  precisions = []
  recalls = []

  for i in thresholds:
    clf._intercept_ = np.asarray([i]) # Modify b
    ys_predicted = clf.predict(xs[TRAIN_SIZE:]) # Predict

    # ys_predicted = clf.decision_function(xs[TRAIN_SIZE:]) # distance not normalized yet
    logging.info("threshold" + str(i))
    logging.info("predicted results:")
    logging.info(ys_predicted)
    logging.info("actual results:")
    logging.info(ys[TRAIN_SIZE:])

    accuracy = clf.score(xs[TRAIN_SIZE:], ys[TRAIN_SIZE:])
    accuracies.append(accuracy)
    precision = metrics.precision_score(ys[TRAIN_SIZE:], ys_predicted)
    precisions.append(precision)
    recall = metrics.recall_score(ys[TRAIN_SIZE:], ys_predicted)
    recalls.append(recall)

    logging.info("acc: {:.2f}, pre: {:.2f}, rec: {:.2f}".format(accuracy, precision, recall))

  plt.clf()
  plt.plot(thresholds, accuracies, label='accuracy', c='r')
  plt.plot(thresholds, precisions, label='precision', c='g')
  plt.plot(thresholds, recalls, label='recall', c='b')
  plt.xlabel('Threshold')
  plt.ylim([0.0, 1.05])
  plt.xlim([-0.2, 0.2])
  plt.title('Accuracy/Precision/Recall')
  plt.legend(loc="lower left")
  plt.show()

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
  clf = svm.SVC(kernel='linear', C=1)
  run_svm(clf, xs, ys)

  # clfw = svm.SVC(kernel='linear', C=1, probability=True, class_weight={1: 10})
  # run_svm(clfw, xs, ys)

if __name__ == '__main__': main()