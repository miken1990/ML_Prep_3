from enum import Enum
import Consts
import pandas as pd
from pandas import read_csv
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


class Modeling:

    def __init__(self):
        self.trainData = None
        self.valData = None
        self.testData = None
        self.trainLabel = None
        self.valLabel = None
        self.testLabel = None

    class ClassifierType(Enum):
        DECISION_TREE = 1

    class ScoreType(Enum):
        ACCURACY = 1
        ERROR = 2
        PRECISION = 3
        RECALL = 4
        FP_RATE = 5
        F1_SCORE = 6 # Harmonic 2/(1/precision + 1/recall)

    def load_data(self, dirPath: str) -> None:
        """
        this method will load ready to use data for the training, validating, and testing sets.
        this implements stages 1, 3 and part of 6 in the assignment.
        :param file_path: the location of the data csv.
        :return:
        """
        # load train features and labels
        trainFileNameX = dirPath.format(Consts.FileSubNames.X_TRAIN.value)
        self._load_data(self.trainData, trainFileNameX)
        trainFileNameY = dirPath.format(Consts.FileSubNames.Y_TRAIN.value)
        self._load_data(self.trainLabel, trainFileNameY)
        # load validation features and labels
        valFileNameX = dirPath.format(Consts.FileSubNames.X__VAL.value)
        self._load_data(self.valData, valFileNameX)
        valFileNameY = dirPath.format(Consts.FileSubNames.Y_VAL.value)
        self._load_data(self.valLabel, valFileNameY)
        # load test features and labels
        testFileNameX = dirPath.format(Consts.FileSubNames.X_TEST.value)
        self._load_data(self.testData, testFileNameX)
        testFileNameY = dirPath.format(Consts.FileSubNames.Y_TEST.value)
        self._load_data(self.testLabel, testFileNameY)

    def _load_data(self, loadedData, filePath):
        loadedData = read_csv(filePath, header=0, keep_default_na=True)

    def createClassifiersNameTuple(self, classifier_type_list: [ClassifierType]):
        """
        this method will train a few classifiers and store them in classifiers list.
        default amount is 2, as required in the assignment.
        :param classifier_type_list: list of classifier names
        saves a list of tuples: (classifier, name) in self.clfName
        """
        self.clfName = []
        if Consts.Classifiers.TREE.value in classifier_type_list:
            decisionTreeClf = DecisionTreeClassifier(criterion='entropy', random_state=Consts.listRandomStates[0],
                                                     max_leaf_nodes=Consts.maxLeafNodes)
            decisionTreeClf.fit(self.trainData, self.trainLabel)
            self.clfName.append((decisionTreeClf, Consts.Classifiers.TREE.value))

        if Consts.Classifiers.SVM.value in classifier_type_list:
            svmClf = SVC(random_state=Consts.listRandomStates[0])
            svmClf.fit(self.trainData, self.trainLabel)
            self.clfName.append((svmClf, Consts.Classifiers.SVM.value))

    def crossValEval(self, scoreMetric: str):
        """
        :param scoreMetric:
        :return:
        """
        self.clfNameScore = []
        for clf, name in self.clfName:
            score = metrics.accuracy_score(self.valLabel, clf.predict(self.valData))
            self.clfNameScore.append(name, score)
        self.bestClfAndName = max(self.clfNameScore, key=lambda x: x[1])


    def trainBestclassifierByTrainAndValidation(self):
        """
        create a classifier from both the train and validate sets of data.
        :param classifier_type: the wanted type of classifier.
        :return:
        """

    def score_classifiers(self, score_type: ScoreType):
        """
        run the validation set on each classifier and give it a score.
        :return: a list of tuples (classifier, score)
        """

    def get_best_models(self, classifiers_score_list: list, n: int = 1):
        """
        get the n best classifiers
        :param classifiers_score_list: a list of tuples (classifier, score)
        :param n: the wanted amount of classifiers
        :return: a list of the n best models
        """

    def classify_test_and_compute_results(self, classifiers_score):
        pass


print(Consts.FileSubNames.X_TRAIN.value)