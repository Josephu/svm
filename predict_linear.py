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
  f1_scores = []

  for i in thresholds:
    clf._intercept_ = np.asarray([i]) # Modify b
    ys_predicted = clf.predict(xs[TRAIN_SIZE:]) # Predict
    # ys_predicted = clf.decision_function(xs[TRAIN_SIZE:]) # distance not normalized yet

    accuracy = clf.score(xs[TRAIN_SIZE:], ys[TRAIN_SIZE:])
    accuracies.append(accuracy)
    f1_score = metrics.f1_score(ys[TRAIN_SIZE:], ys_predicted, average='micro')
    f1_scores.append(f1_score)
    precision = metrics.precision_score(ys[TRAIN_SIZE:], ys_predicted)
    precisions.append(precision)
    recall = metrics.recall_score(ys[TRAIN_SIZE:], ys_predicted)
    recalls.append(recall)

    #logging.info("acc: {:.2f}, pre: {:.2f}, rec: {:.2f}".format(accuracy, precision, recall))

  plt.plot(thresholds, accuracies, label='accuracy', c='r')
  plt.plot(thresholds, f1_scores, label='f1 score', c='y')
  plt.plot(thresholds, precisions, label='precision', c='g')
  plt.plot(thresholds, recalls, label='recall', c='b')
  plt.plot([default_threshold, default_threshold], [0, 1.05], linestyle=':')
  #plt.xlabel('Threshold')
  plt.ylim([0.0, 1.05])
  plt.xlim([start_threshold, end_threshold])
  plt.legend(loc="lower left")

def main():
  if len(sys.argv) < 2:
    print('Please choose input file, eg. python predict_linear.py input.csv')
    exit(0)

  input_file = sys.argv[1]
  c_value = str(C_VALUE)
  dataset = np.genfromtxt(input_file, delimiter=',', skip_header=1)

  ys = dataset[:][:,0]
  xs = dataset[:][:,1:]

  logging.basicConfig(filename='predict.log',level=logging.DEBUG, format='')

  plt.clf()

  # train model
  plt.subplot(2,2,1)
  plt.title('C=1, gamma=0.0')

  clf = svm.SVC(kernel='rbf', C=1, gamma=0, probability=True)
  predict(clf, xs, ys)

  plt.subplot(2,2,2)
  plt.title('C=1, gamma=0.01')

  clf = svm.SVC(kernel='rbf', C=1, gamma=0.01, probability=True)
  predict(clf, xs, ys)

  plt.subplot(2,2,3)
  plt.title('C=1, gamma=0.1')

  clf = svm.SVC(kernel='rbf', C=1, gamma=0.1, probability=True)
  predict(clf, xs, ys)

  plt.subplot(2,2,4)
  plt.title('C=1, gamma=1.0')

  clf = svm.SVC(kernel='rbf', C=1, gamma=1.0, probability=True)
  predict(clf, xs, ys)


  plt.show()


if __name__ == '__main__': main()