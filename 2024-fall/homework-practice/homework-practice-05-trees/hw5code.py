import numpy as np
from collections import Counter
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import bisect


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    sorted_idx = np.argsort(feature_vector)
    feature_sorted = feature_vector[sorted_idx]
    target_sorted = target_vector[sorted_idx]

    filter = np.zeros(len(feature_sorted), dtype=bool)
    filter[0] = True
    filter[1:] = feature_sorted[:-1] != feature_sorted[1:]

    thresholds = (feature_sorted[filter][:-1] + feature_sorted[filter][1:]) / 2

    n = len(feature_sorted)

    cumsum_1 = np.cumsum(target_sorted == 1)
    total_1 = cumsum_1[-1]

    n_L = np.arange(1, n)
    n_R = n - n_L

    n_L_1 = cumsum_1[:-1]
    n_R_1 = total_1 - n_L_1

    p_L_1 = n_L_1 / n_L
    p_L_0 = 1 - p_L_1

    p_R_1 = n_R_1 / n_R
    p_R_0 = 1 - p_R_1

    H_L = 1 - p_L_1**2 - p_L_0**2
    H_R = 1 - p_R_1**2 - p_R_0**2

    ginis = -n_L / n * H_L - n_R / n * H_R

    ginis = ginis[filter[1:]]

    best_idx = np.argmax(ginis) 
    
    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]

    return thresholds, ginis, threshold_best, gini_best

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def set_node(self, node, sub_X, sub_y):
        node["type"] = "terminal"
        node["class"] = Counter(sub_y).most_common(1)[0][0]

    def get_node(self, node, x):
        return node['class']
    
    def find_best_split(self, feature_vector, sub_X, feature, sub_y):
        _, _, threshold, gini = find_best_split(feature_vector, sub_y)
        return threshold, gini

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]) or (self._max_depth and depth >= self._max_depth):
            self.set_node(node, sub_X, sub_y)
            return
        
        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                                    
                sorted_categories = list(map(lambda x: x[1], sorted(ratio.items(), key=lambda x: x[1]))) 
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories))))) 
                feature_vector = np.array(list(map(lambda x: categories_map[ratio[x]], sub_X[:, feature])))
        
            else:
                raise ValueError
            
            if len(set(feature_vector)) <= 1 or len(feature_vector) <= self._min_samples_split:
                continue

            threshold, gini = self.find_best_split(feature_vector, sub_X, feature, sub_y)

            if threshold is None:
                continue

            n_L = len(feature_vector[feature_vector < threshold])
            n_R = len(feature_vector) - n_L

            if self._min_samples_leaf and min(n_L, n_R) < self._min_samples_leaf:
                continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                split = feature_vector < threshold
                gini_best = gini
                threshold_best = threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [x for x in counts.keys() if categories_map[ratio[x]] <= threshold]
                else:
                    raise ValueError

        if feature_best is None:
            self.set_node(node, sub_X, sub_y)
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}

        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return self.get_node(node, x)
        
        if 'categories_split' in node:

            if x[node["feature_split"]] in node['categories_split']:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

        elif "threshold" in node:

            if x[node["feature_split"]] < node['threshold']:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

        else:
            raise ValueError

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)


class LinearRegressionTree(DecisionTree):
    def __init__(self, feature_types, base_model_type=None, max_depth=None, min_samples_split=20, min_samples_leaf=None, n_q=10):
        super().__init__(feature_types, max_depth, min_samples_split, min_samples_leaf)
        self._n_q = n_q

    def set_node(self, node, sub_X, sub_y):
        node["type"] = "terminal"
        node["class"] = Ridge().fit(sub_X, sub_y)

    def get_node(self, node, x):
        return node['class'].predict(x.reshape(1, -11))
    
    def find_best_split(self, feature_vector, sub_X, feature, sub_y):

        sorted_idx = np.argsort(feature_vector)
        feature_sorted = feature_vector[sorted_idx]
        target_sorted = sub_y[sorted_idx]

        X = sub_X.copy()

        X[:, feature] = feature_vector
        sorted_X = sub_X[sorted_idx, :]

        num = self._n_q + 2
        thresholds = list(set(np.percentile(feature_sorted, np.linspace(0, 100, num)[1:-1])))

        n = len(sub_y)
        scores = []

        for th in thresholds:

            idx = bisect.bisect_left(feature_sorted, th)

            n_L = idx
            n_R = n - idx

            if n_L < 1 or n_R < 1:
                continue

            X_L, y_L = sorted_X[:idx, :], target_sorted[:idx]
            X_R, y_R = sorted_X[idx:, :], target_sorted[idx:]

            score_L = mean_squared_error(y_L, Ridge().fit(X_L, y_L).predict(X_L))
            score_R = mean_squared_error(y_R, Ridge().fit(X_R, y_R).predict(X_R))

            score = - n_L / n * score_L - n_R / n * score_R
            scores.append(score)

        if len(scores) == 0:
            return None, None
    
        idx = np.argmax(scores)

        return thresholds[idx], scores[idx]