import sys
import pdb
import logging
import time
import numpy as np
from sklearn import svm, metrics
import matplotlib.pyplot as plt
from pprint import pprint # print beautifully

TRAIN_SIZE = 200

def range_array(start, end, dist):
  thresholds = []
  i = float(start)
  while i <= float(end+dist):
    thresholds.append(float(i))
    i += float(dist)
  return thresholds

def train_and_predict(xs, ys):
  accuracies = []
  f1_scores = []
  cs = range_array(0.05, 10, 0.05)

  for c in cs:
    clf = svm.SVC(kernel='linear', C=c, probability=True)
    clf.fit(xs[:TRAIN_SIZE], ys[:TRAIN_SIZE])

    logging.info("current time %s" % time.strftime("%c") )
    # logging.info("support vectors:")
    # logging.info(clf.support_vectors_)
    # logging.info("number of support vectors for each class: " + str(clf.n_support_))
    # logging.info("threshold: "+ str(clf._intercept_))

    ys_predicted = clf.predict(xs[TRAIN_SIZE:]) # Predict
    # ys_predicted = clf.decision_function(xs[TRAIN_SIZE:]) # distance not normalized yet

    accuracy = clf.score(xs[TRAIN_SIZE:], ys[TRAIN_SIZE:])
    f1_score = metrics.f1_score(ys[TRAIN_SIZE:], ys_predicted, average='micro')
    accuracies.append(accuracy)
    f1_scores.append(f1_score)
  return cs, f1_scores, accuracies

def main():
  if len(sys.argv) < 2:
    print('Please choose input file, eg. python prepare_data.py input.csv')
    exit(0)

  input_file = sys.argv[1]
  dataset = np.genfromtxt(input_file, delimiter=',')

  ys = dataset[:][:,0]
  xs = dataset[:][:,1:]

  logging.basicConfig(filename='optimize.log',level=logging.DEBUG, format='')

  plt.clf()
  plt.title('C-F1 plot')
  cs, f1_scores, accuracies = train_and_predict(xs, ys)

  plt.plot(cs, accuracies, label='acc', c='r')
  plt.plot(cs, f1_scores, label='f1', c='y')
  plt.xlabel('C')
  plt.ylabel('F1')
  plt.ylim([0.0, 1.05])
  plt.xlim([cs[0], cs[-1]])
  plt.legend(loc="lower left")
  plt.show()


if __name__ == '__main__': main()