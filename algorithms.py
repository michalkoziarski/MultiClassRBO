import numpy as np


def distance(x, y, p_norm=1):
    return np.sum(np.abs(x - y) ** p_norm) ** (1 / p_norm)


def rbf(d, gamma):
    if gamma == 0.0:
        return 0.0
    else:
        return np.exp(-(d / gamma) ** 2)


def mutual_class_potential(point, majority_points, minority_points, gamma):
    result = 0.0

    for majority_point in majority_points:
        result += rbf(distance(point, majority_point), gamma)

    for minority_point in minority_points:
        result -= rbf(distance(point, minority_point), gamma)

    return result


def generate_possible_directions(n_dimensions, excluded_direction=None):
    possible_directions = []

    for dimension in range(n_dimensions):
        for sign in [-1, 1]:
            if excluded_direction is None or (excluded_direction[0] != dimension or excluded_direction[1] != sign):
                possible_directions.append((dimension, sign))

    np.random.shuffle(possible_directions)

    return possible_directions


class RBO:
    def __init__(self, gamma=0.05, step_size=0.001, n_steps=500, approximate_potential=True,
                 n_nearest_neighbors=25, minority_class=None, n=None):
        self.gamma = gamma
        self.step_size = step_size
        self.n_steps = n_steps
        self.approximate_potential = approximate_potential
        self.n_nearest_neighbors = n_nearest_neighbors
        self.minority_class = minority_class
        self.n = n

    def fit_sample(self, X, y):
        classes = np.unique(y)

        if self.minority_class is None:
            sizes = [sum(y == c) for c in classes]

            minority_class = classes[np.argmin(sizes)]
        else:
            minority_class = self.minority_class

        minority_points = X[y == minority_class].copy()
        majority_points = X[y != minority_class].copy()

        if self.n is None:
            n = len(majority_points) - len(minority_points)
        else:
            n = self.n

        appended = []
        sorted_neighbors_indices = None
        considered_minority_points_indices = range(len(minority_points))

        n_synthetic_points_per_minority_object = {i: 0 for i in considered_minority_points_indices}

        for _ in range(n):
            idx = np.random.choice(considered_minority_points_indices)
            n_synthetic_points_per_minority_object[idx] += 1

        for i in considered_minority_points_indices:
            if n_synthetic_points_per_minority_object[i] == 0:
                continue

            point = minority_points[i]

            if self.approximate_potential:
                if sorted_neighbors_indices is None:
                    distance_vector = [distance(point, x) for x in X]
                    distance_vector[i] = -np.inf
                    indices = np.argsort(distance_vector)[:(self.n_nearest_neighbors + 1)]
                else:
                    indices = sorted_neighbors_indices[i][:(self.n_nearest_neighbors + 1)]

                closest_points = X[indices]
                closest_labels = y[indices]
                closest_minority_points = closest_points[closest_labels == minority_class]
                closest_majority_points = closest_points[closest_labels != minority_class]
            else:
                closest_minority_points = minority_points
                closest_majority_points = majority_points

            for _ in range(n_synthetic_points_per_minority_object[i]):
                translation = [0 for _ in range(len(point))]
                translation_history = [translation]
                potential = mutual_class_potential(point, closest_majority_points, closest_minority_points, self.gamma)
                possible_directions = generate_possible_directions(len(point))

                for _ in range(self.n_steps):
                    if len(possible_directions) == 0:
                        break

                    dimension, sign = possible_directions.pop()
                    modified_translation = translation.copy()
                    modified_translation[dimension] += sign * self.step_size
                    modified_potential = mutual_class_potential(point + modified_translation, closest_majority_points,
                                                                closest_minority_points, self.gamma)

                    if np.abs(modified_potential) < np.abs(potential):
                        translation = modified_translation
                        translation_history.append(translation)
                        potential = modified_potential
                        possible_directions = generate_possible_directions(len(point), (dimension, -sign))

                appended.append(point + translation)

        return appended


class MultiClassRBO:
    def __init__(self, gamma=0.05, step_size=0.001, n_steps=500, approximate_potential=True,
                 n_nearest_neighbors=25, method='sampling'):
        assert method in ['sampling', 'complete']

        self.gamma = gamma
        self.step_size = step_size
        self.n_steps = n_steps
        self.approximate_potential = approximate_potential
        self.n_nearest_neighbors = n_nearest_neighbors
        self.method = method

    def fit_sample(self, X, y):
        classes = np.unique(y)
        sizes = np.array([float(sum(y == c)) for c in classes])
        indices = np.argsort(sizes)[::-1]
        classes = classes[indices]
        observations = [X[y == c] for c in classes]
        n_max = len(observations[0])

        if self.method == 'sampling':
            for i in range(1, len(classes)):
                cls = classes[i]
                n = n_max - len(observations[i])
                X_sample = [observations[i]]
                y_sample = [cls * np.ones(len(observations[i]))]

                for j in range(0, i):
                    indices = np.random.choice(range(len(observations[j])), int(n_max / i))
                    X_sample.append(observations[j][indices])
                    y_sample.append(classes[j] * np.ones(len(X_sample[-1])))

                oversampler = RBO(gamma=self.gamma, step_size=self.step_size, n_steps=self.n_steps,
                                  approximate_potential=self.approximate_potential,
                                  n_nearest_neighbors=self.n_nearest_neighbors, minority_class=cls, n=n)

                appended = oversampler.fit_sample(np.concatenate(X_sample), np.concatenate(y_sample))

                if len(appended) > 0:
                    observations[i] = np.concatenate([observations[i], appended])
        else:
            for i in range(1, len(classes)):
                cls = classes[i]
                n = n_max - len(observations[i])

                oversampler = RBO(gamma=self.gamma, step_size=self.step_size, n_steps=self.n_steps,
                                  approximate_potential=self.approximate_potential,
                                  n_nearest_neighbors=self.n_nearest_neighbors, minority_class=cls, n=n)

                appended = oversampler.fit_sample(X, y)

                if len(appended) > 0:
                    observations[i] = np.concatenate([observations[i], appended])

        labels = [cls * np.ones(len(obs)) for obs, cls in zip(observations, classes)]

        return np.concatenate(observations), np.concatenate(labels)
