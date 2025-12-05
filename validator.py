# validator.py
import numpy as np

class Validator:
    def __init__(self, classifier):
        self.classifier = classifier

    def leave_one_out_accuracy(self, X, y, features):
        
        cols = [f - 1 for f in features]
        X = X[:, cols]
        correct = 0
        n = len(X)

        for i in range(n):
            X_train = np.delete(X, i, axis = 0)
            y_train = np.delete(y, i, axis = 0)
            X_test = X[i]

            self.classifier.datastore(X_train, y_train)
            pred = self.classifier.test(X_test)

            if pred == y[i]:
                correct += 1

        return 100.0 * correct / n