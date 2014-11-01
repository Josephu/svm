import sys
import pdb
import logging
import time
import numpy as np
from sklearn import svm, metrics
import matplotlib.pyplot as plt
from pprint import pprint # print beautifully

TRAIN_SIZE = 200
C_VALUE = 1

def threshold_array(start, end, dist):
  thresholds = []
  i = float(start)
  while i <= float(end+dist):
    thresholds.append(float(i))
    i += float(dist)
  return thresholds

def predict(clf, xs, ys):
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

  default_threshold = clf._intercept_
  start_threshold = float(default_threshold - 3)
  end_threshold = float(default_threshold + 3)
  thresholds = threshold_array(start_threshold, end_threshold, 0.1)
  accuracies = []
  precisions = []
  recalls = []

  for i in thresholds:
    clf._intercept_ = np.asarray([i]) # Modify b
    ys_predicted = clf.predict(xs[TRAIN_SIZE:]) # Predict
    # ys_predicted = clf.decision_function(xs[TRAIN_SIZE:]) # distance not normalized yet

    accuracy = clf.score(xs[TRAIN_SIZE:], ys[TRAIN_SIZE:])
    accuracies.append(accuracy)
    precision = metrics.precision_score(ys[TRAIN_SIZE:], ys_predicted)
    precisions.append(precision)
    recall = metrics.recall_score(ys[TRAIN_SIZE:], ys_predicted)
    recalls.append(recall)

    logging.info("acc: {:.2f}, pre: {:.2f}, rec: {:.2f}".format(accuracy, precision, recall))

  return accuracies, precisions, recalls, thresholds, default_threshold, start_threshold, end_threshold

def main():
  if len(sys.argv) < 2:
    print('Please choose input file, eg. python prepare_data.py input.csv')
    exit(0)

  input_file = sys.argv[1]
  c_value = str(C_VALUE)
  dataset = np.genfromtxt(input_file, delimiter=',', skip_header=1)

  ys = dataset[:][:,0]
  xs = dataset[:][:,1:]

  logging.basicConfig(filename='predict.log',level=logging.DEBUG, format='')

  # train model
  clf = svm.SVC(kernel='linear', C=1, probability=True)
  acc1, prec1, rec1, thres1, default_threshold1, start_threshold1, end_threshold1 = predict(clf, xs, ys)

  clfw = svm.SVC(kernel='linear', C=1, probability=True, class_weight={1: 3})
  acc2, prec2, rec2, thres2, default_threshold2, start_threshold2, end_threshold2 = predict(clfw, xs, ys)

  plt.clf()
  plt.subplot(2,2,1)
  plt.title('No weight')
  plt.plot(thres1, acc1, label='accuracy', c='r')
  plt.plot(thres1, prec1, label='precision', c='g')
  plt.plot(thres1, rec1, label='recall', c='b')
  plt.plot([default_threshold1, default_threshold1], [0, 1.05], linestyle=':')
  plt.xlabel('Threshold')
  plt.ylim([0.0, 1.05])
  plt.xlim([start_threshold1, end_threshold1])
  plt.legend(loc="lower left")
  plt.subplot(2,2,2)
  plt.title('Weight 1:3')
  plt.plot(thres2, acc2, label='accuracy', c='r')
  plt.plot(thres2, prec2, label='precision', c='g')
  plt.plot(thres2, rec2, label='recall', c='b')
  plt.plot([default_threshold2, default_threshold2], [0, 1.05], linestyle=':')
  plt.xlabel('Threshold')
  plt.ylim([0.0, 1.05])
  plt.xlim([start_threshold2, end_threshold2])
  plt.legend(loc="lower left")

  plt.show()


if __name__ == '__main__': main()