import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from mlxtend.classifier import Adaline

from LMS import Lms

class CompareModels:
    def __init__(self):
        self.heldout = [0.85, 0.80, 0.70, 0.60, 0.50, 0.40, 0.01]
        self.rounds = 20
        self.digits = datasets.load_digits()
        self.digits_X, self.digits_y = self.digits.data, self.digits.target
        self.iris = datasets.load_iris()
        self.iris_X, self.iris_y = self.iris.data, self.iris.target

        self.classifiers = [
            ("Perceptron", Perceptron(max_iter=30)),
            ("Adaline", Adaline(epochs=30,
              eta=0.01,
              minibatches=1,
              random_seed=1)),
            ("Lms", Lms(eta=0.01, n_iter=30, random_state=1))
        ]
        self.xx = 1. - np.array(self.heldout)

    def compareDigitsBinary1VsAll(self):
        self._compareDatasetBinary1VsAll(self.digits_X, self.digits_y, "Digits")

    def compareIrisBinary1VsAll(self):
        self._compareDatasetBinary1VsAll(self.iris_X, self.iris_y, "Iris")

    def _compareDatasetBinary1VsAll(self, X, y, sTitle):
        lb = preprocessing.LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
        lb.fit(y)
        oneVsAll_Y = lb.transform(y)
        print(oneVsAll_Y.shape)
        for name, clf in self.classifiers:
            print("training %s" % name)
            rng = np.random.RandomState(42)
            yy = []
            for i in self.heldout:
                yy_ = []
                for r in range(self.rounds):
                    yy__ = []
                    for col in range(oneVsAll_Y.shape[1]):
                        X_train, X_test, y_train, y_test = \
                            train_test_split(X, oneVsAll_Y[:, [col]].ravel(), test_size=i, random_state=rng)
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        yy__.append(1 - np.mean(y_pred == y_test))
                    yy_.append(np.mean(yy__))
                yy.append(np.mean(yy_))
            plt.plot(self.xx, yy, label=name)

        plt.title(sTitle)
        plt.legend(loc="upper right")
        plt.xlabel("Proportion train")
        plt.ylabel("Test Error Rate")
        plt.show()

cmp = CompareModels()
cmp.compareDigitsBinary1VsAll()
cmp.compareIrisBinary1VsAll()