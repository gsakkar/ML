
from decision_tree import MyDecisionTree
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt


# Load the breast cancer dataset
data = load_breast_cancer()
x, y = data.data, data.target


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


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
        self.history = {'trees': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    def fit(self, x, y, x_val=None, y_val=None):
        self.trees = []
        for i in range(self.n_trees):
            tree = MyDecisionTree()
            x_samp, y_samp = bootstrap_sample(x, y)
            tree.fit(x_samp, y_samp)
            self.trees.append(tree)

             # Calculate and store metrics after adding each tree
            if x_val is not None and y_val is not None:
                y_pred = self.predict(x_val)
                self.history['trees'].append(i + 1)
                self.history['accuracy'].append(accuracy_score(y_val, y_pred))
                self.history['precision'].append(precision_score(y_val, y_pred, average='weighted'))
                self.history['recall'].append(recall_score(y_val, y_pred, average='weighted'))
                self.history['f1'].append(f1_score(y_val, y_pred, average='weighted'))
            
            # Print progress
            if (i + 1) % 10 == 0 or i + 1 == self.n_trees:
                print(f"Tree {i + 1}/{self.n_trees} built")

    def predict(self, x):
        tree_preds = np.array([tree.predict(x) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)

    def compile(self, loss, optimizer):
        # This method is added for compatibility with the Keras API
        # but doesn't do anything for our RandomForest implementation
        pass

    def evaluate(self, x, y):
        # Implement evaluation logic
        y_pred = self.predict(x)
        return np.mean((y - y_pred) ** 2)  # MSE as an example

    def accuracy(self, y_true, y_pred):
        # Implement accuracy calculation
        return np.mean(y_true == y_pred)
    
    def plot_performance_history(self):
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        plt.figure(figsize=(15, 10))
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 2, i)
            plt.plot(self.history['trees'], self.history[metric])
            plt.title(f'Random Forest {metric.capitalize()}')
            plt.xlabel('Number of Trees')
            plt.ylabel(metric.capitalize())
        
        plt.tight_layout()
        plt.show()


# Create an instance of the modified Random forest model
model = RandomForest(n_trees=3, max_depth=10)

# Compile the model (this doesn't do anything but is kept for API compatibility)
model.compile(loss='mse', optimizer='adam')

# Train the model
model.fit(x_train, y_train)

# Evaluate the model
loss = model.evaluate(x_test, y_test)

# Make predictions using the trained model
y_pred = model.predict(x_test)

# Calculate the accuracy of the predictions
accuracy = model.accuracy(y_test, y_pred)