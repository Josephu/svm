import pdb, csv
import numpy as np
from sklearn import preprocessing

dataset = np.genfromtxt('wholesale_customers.csv', delimiter=',', skip_header=1)
np.random.shuffle(dataset)

min_max_scaler = preprocessing.MinMaxScaler()
xs_scale = min_max_scaler.fit_transform(dataset[:][:,2:])
ys = dataset[:][:,0]

ys[ys == 1] = -1
ys[ys == 2] = 1

dataset_processed = np.c_[ys, xs_scale]

np.savetxt("wholesale_customers_processed.csv", dataset_processed, delimiter=",")