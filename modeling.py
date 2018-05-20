from enum import Enum

from sklearn.ensemble import RandomForestClassifier

import Consts
import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, RandomizedSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC



class Modeling:
    dict_dfs = {d: None for d in list(Consts.FileSubNames)}

    def __init__(self):
        self.dict_dfs = {d: None for d in list(Consts.FileSubNames)}

    class ClassifierType(Enum):
        DECISION_TREE = 'decision_tree'
        SVM = 'svm'
        RANDOM_FOREST = 'random forest'

    class ScoreType(Enum):
        # Classification
        F1= 'f1'
        F1_MACRO= 'f1_macro'
        F1_MICRO= 'f1_micro'
        F1_WEIGHTED = 'f1_weighted'
        ACCURACY = 'accuracy'
        # Clustering

        # Regression
        EXPLAINED_VARIANCE = 'explained_variance'
        R2 = 'r2'

    def load_data(self, base: Consts.FileNames, set: int) -> None:
        """
        this method will load ready to use data for the training, validating, and testing sets.
        this implements stages 1, 3 and part of 6 in the assignment.
        :param file_path: the location of the data csv.
        :return:
        """
        # load train features and labels
        for d in list(Consts.FileSubNames):
            file_location = base.value.format(set, d.value)
            if d in {Consts.FileSubNames.Y_TEST, Consts.FileSubNames.Y_VAL, Consts.FileSubNames.Y_TRAIN }:
                self.dict_dfs[d] = self._load_data(file_location)[Consts.VOTE_STR].as_matrix().ravel()
            else:
                self.dict_dfs[d] = self._load_data(file_location).as_matrix()

    def _load_data(self, filePath):
        return read_csv(filePath, header=0, keep_default_na=True)

    def parameter_search_classifiers(self, classifier_types: Consts.ClassifierType=list(Consts.ClassifierType),
                                     scoring: Consts.ScoreType=Consts.ScoreType.ACCURACY) -> list:
        list_random_search = []     # type: [RandomizedSearchCV]

        if Consts.ClassifierType.DECISION_TREE in classifier_types:
            clf = DecisionTreeClassifier()

        if Consts.ClassifierTypes.SVM in classifier_types:
            clf = SVC()

        if Consts.ClassifierTypes.RANDOM_FOREST in classifier_types:
            clf = RandomForestClassifier()

        [random_search.fit(self.dict_dfs[Consts.FileSubNames.X_TRAIN], self.dict_dfs[Consts.FileSubNames.Y_TRAIN])
         for random_search in list_random_search]

        return list_random_search

    def apply_trained_models_on_validation(self, list_estimators: list, scoring: Consts.ScoreType ):

        for model in list_estimators:
            score = cross_val_score(model, self.dict_dfs[Consts.FileSubNames.X_VAL],
                                    self.dict_dfs[Consts.FileSubNames.Y_VAL],
                                    scoring=scoring.value)
def ex_3():
    m = Modeling()

    # Use set 1
    set = 1

    # load the data from set 1.
    m.load_data(Consts.FileNames.FILTERED_AND_SCALED, set)

    # search for good parameters by using cross val.
    list_random_search = m.parameter_search_classifiers()





#
#
#     def create_classifiers_name_tuple(self, classifier_type_list: [ClassifierType]):
#         """
#         this method will train a few classifiers and store them in classifiers list.
#         :param classifier_type_list: list of classifiers by Enum
#         saves a list of tuples: (classifier obj, classifier Enum) in self.clfName
#         """
#         self.list_clf_and_type = []
#         if Consts.ClassifierTypes.TREE in classifier_type_list:
#             decisionTreeClf = DecisionTreeClassifier(criterion='entropy', random_state=Consts.listRandomStates[0],
#                                                      max_leaf_nodes=Consts.maxLeafNodes)
#             decisionTreeClf.fit(self.train_data, self.train_label)
#             self.list_clf_and_type.append((decisionTreeClf, Consts.ClassifierTypes.TREE))
#
#         if Consts.ClassifierTypes.SVM in classifier_type_list:
#             svmClf = SVC(random_state=Consts.listRandomStates[0])
#             svmClf.fit(self.train_data, self.train_label)
#             self.list_clf_and_type.append((svmClf, Consts.ClassifierTypes.SVM))
#
#         #TODO: add more classifiers
#
#     def cross_val_eval(self, scoreMetric: str=None):
#         """
#         :param scoreMetric:
#         :return:
#         """
#         for clf, name in self.list_clf_and_type:
#             #TODO: use a callable func to score the classifiers differently?
#             score = metrics.accuracy_score(self.val_label, clf.predict(self.val_data))
#             self.list_name_score.append((name, score))
#
#     def train_best_classifier_by_train_and_validation(self):
#         """
#         create a classifier from both the train and validate sets of data.
#         :param classifier_type: the wanted type of classifier.
#         :return:
#         """
#         self.get_best_models()
#         X_data, Y_data = self.concat_train_and_val()
#         clf = None
#         if self.best_name_and_score[0] == Consts.ClassifierTypes.TREE:
#             clf = DecisionTreeClassifier(criterion='entropy', random_state=Consts.listRandomStates[0],
#                                          max_leaf_nodes=Consts.maxLeafNodes)
#         elif self.best_name_and_score[0] == Consts.ClassifierTypes.SVM:
#             clf = SVC(random_state=Consts.listRandomStates[0])
#         #TODO: if more classifiers are added, they should be trained here.
#
#         clf.fit(X=X_data, y=Y_data.ravel())
#
#         return clf
#
#     def get_best_models(self):
#         """
#         get the n best classifiers
#         :param classifiers_score_list: a list of tuples (classifier, score)
#         :param n: the wanted amount of classifiers
#         :return: a list of the n best models
#         """
#         self.best_name_and_score = max(self.list_name_score, key=lambda x: x[1])
#         print(f"Choose {self.best_name_and_score[0].value} as the best Classifier")
#
#
#     def classify_test_and_compute_results(self):
#         clf = self.train_best_classifier_by_train_and_validation()
#
#         Y_pred = clf.predict(X=self.test_data)
#
#         print(Y_pred)
#
#
#     def concat_train_and_val(self) -> (pd.DataFrame, pd.DataFrame):
#         return np.concatenate((self.train_data, self.val_data), axis=0), \
#                np.ravel([self.train_label, self.val_label])
#
# if __name__ == "__main__":
#
#     m = Modeling()
#
#     # load the data from set 1.
#     m.load_data(Consts.FileNames.FILTERED_AND_SCALED.value.format(1, "{}"))
#
#     # Save a list of tuples inside m. train the classifiers by the train data.
#     m.create_classifiers_name_tuple(list(Consts.ClassifierTypes))
#
#     # Set a score for each type of classifier. TODO: might change to receive a scoring function
#     m.cross_val_eval()
#
#     # Train the classifier with the highest score and get the results
#     # Here the training is by the train and validation set.
#     m.classify_test_and_compute_results()

