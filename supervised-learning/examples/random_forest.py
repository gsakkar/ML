
from .decision_tree import MyDecisionTree
from collections import Counter
import numpy as np
from tensorflow.keras.datasets import breast_cancer
# Load the breast cancer dataset
(x_train, y_train), (x_test, y_test) = breast_cancer.load_data()


def bootstrap_sample(x, y):
    n_samples = x.shape[0]
    rng = np.random.default_rng()
    idxs = rng.integers(0, n_samples, size=n_samples, dtype=np.int64)
    return x[idxs], y[idxs]


def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common


class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, x, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = MyDecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_feats=self.n_feats,
            )
            x_samp, y_samp = bootstrap_sample(x, y)
            tree.fit(x_samp, y_samp)
            self.trees.append(tree)

    def predict(self, x):
        tree_preds = np.array([tree.predict(x) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)

# Create an instance of the modified Random forest model
model = RandomForest(n_trees=3, max_depth=10)

# Compile and train the model
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train)

# Evaluate the model
loss = model.evaluate(x_test, y_test)

# Make predictions using the trained model
y_pred = model.predict(x_test)

# Calculate the accuracy of the predictions
accuracy = model.accuracy(y_test, y_pred)



