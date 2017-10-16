import numpy as np


def _rbf(d, eps):
    return np.exp(-(d * eps) ** 2)


def _distance(x, y):
    return np.sum(np.abs(x - y))


def _pairwise_distances(X):
    D = np.zeros((len(X), len(X)))

    for i in range(len(X)):
        for j in range(len(X)):
            if i == j:
                continue

            d = _distance(X[i], X[j])

            D[i][j] = d
            D[j][i] = d

    return D


def _score(point, X, y, minority_class, epsilon, cost):
    mutual_density_score = 0.0

    for i in range(len(X)):
        if type(cost[cost.keys()[0]]) is dict:
            current_cost = cost[y[i]][minority_class]
        else:
            current_cost = cost[y[i]]

        rbf = _rbf(_distance(point, X[i]), epsilon / current_cost)

        if y[i] == minority_class:
            mutual_density_score -= rbf
        else:
            mutual_density_score += rbf

    return mutual_density_score


class RBO:
    def __init__(self, gamma=0.05, n_steps=500, step_size=0.001, stop_probability=0.02, criterion='balance',
                 cost=None, minority_class=None, n=None):
        assert criterion in ['balance', 'minimize', 'maximize']
        assert 0.0 <= stop_probability <= 1.0

        self.gamma = gamma
        self.n_steps = n_steps
        self.step_size = step_size
        self.stop_probability = stop_probability
        self.criterion = criterion
        self.cost = cost
        self.minority_class = minority_class
        self.n = n

    def fit_sample(self, X, y):
        epsilon = 1.0 / self.gamma
        classes = np.unique(y)

        if self.cost is None:
            cost = {}

            for current_class in classes:
                cost[current_class] = {}

                for opposing_class in classes:
                    cost[current_class][opposing_class] = 1.0
        else:
            cost = self.cost

        if self.minority_class is None:
            sizes = [sum(y == c) for c in classes]
            minority_class = classes[np.argmin(sizes)]
        else:
            minority_class = self.minority_class

        minority_points = X[y == minority_class]

        if self.n is None:
            n = sum(y != minority_class) - sum(y == minority_class)
        else:
            n = self.n

        minority_scores = []

        for i in range(len(minority_points)):
            minority_point = minority_points[i]
            minority_scores.append(_score(minority_point, X, y, minority_class, epsilon, cost))

        appended = []

        while len(appended) < n:
            idx = np.random.choice(range(len(minority_points)))
            point = minority_points[idx].copy()
            score = minority_scores[idx]

            for i in range(self.n_steps):
                if self.stop_probability is not None and self.stop_probability > np.random.rand():
                    break

                translation = np.zeros(len(point))
                sign = np.random.choice([-1, 1])
                translation[np.random.choice(range(len(point)))] = sign * self.step_size
                translated_point = point + translation
                translated_score = _score(translated_point, X, y, minority_class, epsilon, cost)

                if (self.criterion == 'balance' and np.abs(translated_score) < np.abs(score)) or \
                        (self.criterion == 'minimize' and translated_score < score) or \
                        (self.criterion == 'maximize' and translated_score > score):
                    point = translated_point
                    score = translated_score

            appended.append(point)

        return appended


class MultiClassRBO:
    def __init__(self, gamma=0.05, n_steps=500, step_size=0.001, stop_probability=0.02, criterion='balance',
                 cost=None, method='individual'):
        assert criterion in ['balance', 'minimize', 'maximize']
        assert method in ['individual', 'joint']
        assert 0.0 <= stop_probability <= 1.0

        self.gamma = gamma
        self.n_steps = n_steps
        self.step_size = step_size
        self.stop_probability = stop_probability
        self.criterion = criterion
        self.cost = cost
        self.method = method

    def fit_sample(self, X, y):
        classes = np.unique(y)
        sizes = np.array([float(sum(y == c)) for c in classes])
        indices = np.argsort(sizes)[::-1]
        classes = classes[indices]
        sizes = sizes[indices]
        observations = [X[y == c] for c in classes]
        n_max = len(observations[0])

        if self.cost is None or self.cost is True:
            cost = {}

            for current_class, current_size in zip(classes, sizes):
                cost[current_class] = {}

                for opposing_class, opposing_size in zip(classes, sizes):
                    if self.cost is None:
                        cost[current_class][opposing_class] = 1.0
                    else:
                        cost[current_class][opposing_class] = np.sqrt(opposing_size / current_size)
        else:
            cost = self.cost

        if self.method == 'individual':
            for i in range(1, len(classes)):
                cls = classes[i]
                n = n_max - len(observations[i])
                X_sample = [observations[i]]
                y_sample = [cls * np.ones(len(observations[i]))]

                for j in range(0, i):
                    indices = np.random.choice(range(len(observations[j])), int(n_max / i))
                    X_sample.append(observations[j][indices])
                    y_sample.append(classes[j] * np.ones(len(X_sample[-1])))

                oversampler = RBO(gamma=self.gamma, n_steps=self.n_steps, step_size=self.step_size,
                                  stop_probability=self.stop_probability, criterion=self.criterion,
                                  cost=cost, minority_class=cls, n=n)

                appended = oversampler.fit_sample(np.concatenate(X_sample), np.concatenate(y_sample))

                if len(appended) > 0:
                    observations[i] = np.concatenate([observations[i], appended])
        else:
            avg_cost = cost.copy()

            for k in avg_cost.keys():
                avg_cost[k] = np.mean(avg_cost[k].values())

            for i in range(1, len(classes)):
                cls = classes[i]
                n = n_max - len(observations[i])

                oversampler = RBO(gamma=self.gamma, n_steps=self.n_steps, step_size=self.step_size,
                                  stop_probability=self.stop_probability, criterion=self.criterion,
                                  cost=avg_cost, minority_class=cls, n=n)

                appended = oversampler.fit_sample(X, y)

                if len(appended) > 0:
                    observations[i] = np.concatenate([observations[i], appended])

        labels = [cls * np.ones(len(obs)) for obs, cls in zip(observations, classes)]

        return np.concatenate(observations), np.concatenate(labels)