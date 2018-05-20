import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from mlxtend.classifier import Adaline

from LMS import Lms

class CompareModels:
    def __init__(self, max_iterations):
        self.heldout = [0.85, 0.80, 0.70, 0.60, 0.50, 0.40, 0.01]
        self.rounds = 20
        self.digits = datasets.load_digits()
        self.digits_X, self.digits_y = self.digits.data, self.digits.target
        self.iris = datasets.load_iris()
        self.iris_X, self.iris_y = self.iris.data, self.iris.target
        self.classifiers = [
            ("Perceptron", Perceptron(max_iter=max_iterations, random_state=1, eta0=0.001)),
            # ("Adaline", Adaline(epochs=max_iterations,
            #   eta=0.001,
            #   minibatches=1,
            #   random_seed=1)),
            ("Lms", Lms(eta=0.001, n_iter=max_iterations, random_state=1))
        ]
        self.xx = 1. - np.array(self.heldout)

    def compareDigitsBinary1VsAll(self):
        self._compareDatasetBinary1VsAll(self.digits_X, self.digits_y, "Digits")

    def compareIrisBinary1VsAll(self):
        self._compareDatasetBinary1VsAll(self.iris_X, self.iris_y, "Iris")

    def compareWineBinary1VsAll(self):
        self._compareDatasetBinary1VsAll(self.wine_X, self.wine_y, "Wine")

    def compareBreastCancerBinary1VsAll(self):
        self._compareDatasetBinary1VsAll(self.breast_X, self.breast_y, "Breast Cancer")

    def _compareDatasetBinary1VsAll(self, X, y, sTitle):
        lb = preprocessing.LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
        lb.fit(y)
        oneVsAll_Y = lb.transform(y)
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

    def createLinearySeperableDataset(self, rand_state=0):
        separable = False
        samples = None
        while not separable:
            samples = datasets.make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=1,
                                                   n_clusters_per_class=1, flip_y=-1, random_state=rand_state)
            red = samples[0][samples[1] == 0]
            blue = samples[0][samples[1] == 1]
            separable = any(
                [red[:, k].max() < blue[:, k].min() or red[:, k].min() > blue[:, k].max() for k in range(2)])
            rand_state = rand_state + 1 if not separable else rand_state
        print(rand_state)
        return samples

    def plotDataset(self, samples):
        red = samples[0][samples[1] == 0]
        blue = samples[0][samples[1] == 1]
        plt.plot(red[:, 0], red[:, 1], 'r.')
        plt.plot(blue[:, 0], blue[:, 1], 'b.')
        plt.show()


if __name__ == '__main__':
    cmp = CompareModels(100)
    # compare Iris and Digits
    cmp.compareDigitsBinary1VsAll()
    cmp.compareIrisBinary1VsAll()
    # create dataset where LMS converges faster than Perceptron
    firstSamples = cmp.createLinearySeperableDataset(31)
    first_X, first_y = firstSamples
    cmp._compareDatasetBinary1VsAll(first_X, first_y, 'first')
    cmp.plotDataset(firstSamples)
    # create dataset where Perceptron converges faster than LMS
    secondSamples = cmp.createLinearySeperableDataset(26)
    second_X, second_y = secondSamples
    cmp._compareDatasetBinary1VsAll(second_X, second_y, 'second')
    cmp.plotDataset(secondSamples)
