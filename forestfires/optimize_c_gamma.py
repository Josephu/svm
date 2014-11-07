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

def power_array(start, end, dist):
  math.pow(x, y)

def train_and_predict(xs, ys):
  accuracies = []
  f1_scores = []
  cs_array = []
  gs_array = []
  cs = range_array(-2, 2, 0.1)
  gammas = range_array(-2, 2, 0.1)

  # Need to build 2D array for X, Y, Z
  for c in cs:
    powc = np.power(10,c)
    f1_scores.append([])
    accuracies.append([])
    cs_array.append([])
    gs_array.append([])
    for g in gammas:
      powg = np.power(10,g)

      clf = svm.SVC(kernel='rbf', C=powc, gamma=powg)
      clf.fit(xs[:TRAIN_SIZE], ys[:TRAIN_SIZE])

      ys_predicted = clf.predict(xs[TRAIN_SIZE:]) # Predict

      accuracy = clf.score(xs[TRAIN_SIZE:], ys[TRAIN_SIZE:])
      #f1_score = metrics.f1_score(ys[TRAIN_SIZE:], ys_predicted, average='micro')
      accuracies[-1].append(accuracy)
      cs_array[-1].append(c)
      gs_array[-1].append(g)
      #f1_scores[-1].append(f1_score)
  return cs_array, gs_array, accuracies

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
  cs, gammas, f1_scores = train_and_predict(xs, ys)
  plt.title('C-Gamma-Accuracy Plot')
  plt.xlabel('C - 10.Power(X)')
  plt.ylabel('Gamma - 10.Power(Y)')
  plot = plt.contour(cs, gammas, f1_scores, 8)
  plt.clabel(plot, inline=1, fontsize=9)
  plt.show()


if __name__ == '__main__': main()