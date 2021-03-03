"""
Utils
-----

A few useful functions
"""
import numpy as np

def compute_acc(model, X_test, y_test):
    """Computes a model's accuracy based on test labelled data

    :param model: the model to evaluate
    :param X_test: test set, without labels
    :param y_test: labels
    :return: the estimated accuracy of the model
    :rtype: float
    """
    pred = list(map(np.argmax, model(X_test)))
    true_values = list(map(np.argmax, y_test))
    acc = np.sum(
        [1 for i in range(len(pred)) if pred[i] == true_values[i]]
    ) / len(pred)
    return acc
