import os
import argparse
import algorithms
import datasets
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('-dataset', choices=datasets.names(), required=True)
parser.add_argument('-partition', choices=[1, 2, 3, 4, 5], required=True, type=int)
parser.add_argument('-fold', choices=[1, 2], required=True, type=int)

args = parser.parse_args()

algorithm = algorithms.MultiClassRBO()

(X_train, y_train), (X_test, y_test) = datasets.load(args.dataset, args.partition, args.fold)

X_train, y_train = algorithm.fit_sample(X_train, y_train)

results_path = os.path.join(os.path.dirname(__file__), 'results')
dataset_path = os.path.join(results_path, args.dataset)

for path in [results_path, dataset_path]:
    if not os.path.exists(path):
        os.mkdir(path)

output_file_name = '%s.%d.%d.train.oversampled.csv' % (args.dataset, args.partition, args.fold)
output_path = os.path.join(dataset_path, output_file_name)
original_file_name = output_file_name.replace('.oversampled', '')
original_path = os.path.join(os.path.join(os.path.dirname(__file__), 'data', 'folds', args.dataset, original_file_name))
header = pd.read_csv(original_path).columns
pd.DataFrame(np.c_[X_train, y_train]).to_csv(output_path, index=False, header=header)
