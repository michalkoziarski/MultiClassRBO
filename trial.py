import os
import argparse
import algorithms
import datasets
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('-dataset', choices=datasets.names(), required=True)
parser.add_argument('-partition', choices=[1, 2, 3, 4, 5], type=int, required=True)
parser.add_argument('-fold', choices=[1, 2], type=int, required=True)
parser.add_argument('-mode', choices=['OVA', 'OVO'], default='OVA')
parser.add_argument('-method', choices=['sampling', 'complete'], default='sampling')
parser.add_argument('-results_path', type=str, default=os.path.join(os.path.dirname(__file__), 'results'))

args = parser.parse_args()

dataset_path = os.path.join(args.results_path, args.dataset)

for path in [args.results_path, dataset_path]:
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError:
            print('WARNING: failed to create a directory under "%s", likely due to race condition.' % path)

(X_train, y_train), (X_test, y_test) = datasets.load(args.dataset, args.partition, args.fold)

header = pd.read_csv(
    os.path.join(
        os.path.dirname(__file__),
        'data',
        'folds',
        args.dataset,
        '%s.%d.%d.train.csv' % (args.dataset, args.partition, args.fold)
    )
).columns

if args.mode == 'OVA':
    X_train, y_train = algorithms.MultiClassRBO(method=args.method).fit_sample(X_train, y_train)

    output_path = os.path.join(dataset_path, '%s.%d.%d.train.oversampled.csv' % (
        args.dataset, args.partition, args.fold
    ))

    pd.DataFrame(np.c_[X_train, y_train]).to_csv(output_path, index=False, header=header)
elif args.mode == 'OVO':
    classes = np.unique(np.concatenate([y_train, y_test]))

    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            indices = ((y_train == classes[i]) | (y_train == classes[j]))

            X, y = X_train[indices].copy(), y_train[indices].copy()

            class_distribution = {cls: len(y[y == cls]) for cls in [classes[i], classes[j]]}
            minority_class = min(class_distribution, key=class_distribution.get)

            appended = algorithms.RBO().fit_sample(X, y)

            if len(appended) > 0:
                X, y = np.concatenate([X, appended]), np.concatenate([y, np.tile([minority_class], len(appended))])

            output_path = os.path.join(dataset_path, '%s.%d.%d.train.oversampled.%dv%d.csv' % (
                args.dataset, args.partition, args.fold, classes[i], classes[j]
            ))

            pd.DataFrame(np.c_[X, y]).to_csv(output_path, index=False, header=header)
else:
    raise NotImplementedError
