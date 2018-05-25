from enum import Enum
from time import time
from typing import List

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer

import Consts
import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, RandomizedSearchCV, cross_val_score, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

"""
TODO list:
    1. concatenate_train_and_val - Alon
    2. function receives an estimator and predicts the winner (from test)
    3. function receives an estimator and predicts the distribution between predictions (from test)
    4. function receives an estimator and predicts a list of most likely to vote for each party
        a. from fresh test set.
        b. save each list to a separate csv. (file names in new directory in Consts!)
    5. confusion matrix
    6. compare scoring functions (in a function) and save outputs to csv ot print. save the wanted scoring method.
        - search the web for their plots.
        - explain why they are good for us.
    7. create directories.
    
    if there is time:
    1. plot learning graphs - capture a picture to explain why we choose to concatenate the train and validation set.
    2. plot validation graphs for each parm - we can see how params modify the accuracy, set the distribution 
       accordingly.
"""

def f1_score_not_binary(y_true, y_pred):
    return np.mean(f1_score(y_true, y_pred, average=None))

class Modeling:
    dict_dfs = {d: None for d in list(Consts.FileSubNames)}
    do_print = True

    def __init__(self):
        self.dict_dfs = {d: None for d in list(Consts.FileSubNames)}

    def log(self, msg):
        if self.do_print:
            print(msg)

    def load_data(self, base: Consts.FileNames, set: int) -> None:
        """
        this method will load ready to use data for the training, validating, and testing sets.
        this implements stages 1, 3 and part of 6 in the assignment.
        :param file_path: the location of the data csv.
        :return:
        """
        self.log(f"Loading the data from {base}")
        # load train features and labels
        for d in list(Consts.FileSubNames):
            file_location = base.value.format(set, d.value)
            if d in {Consts.FileSubNames.Y_TEST, Consts.FileSubNames.Y_VAL, Consts.FileSubNames.Y_TRAIN }:
                self.dict_dfs[d] = self._load_data(file_location)[Consts.VOTE_STR].as_matrix().ravel()
            else:
                self.dict_dfs[d] = self._load_data(file_location).as_matrix()

    def _load_data(self, filePath):
        self.log(f"Loading {filePath}")
        return read_csv(filePath, header=0, keep_default_na=True)

    def allocate_rand_search_classifiers(self, classifier_types: {Consts.ClassifierType},
                                         scoring: Consts.ScoreType) -> [RandomizedSearchCV]:
        list_random_search = []  # type: [RandomizedSearchCV]
        n_iter = 1
        n_jobs = 1
        cv = 3
        score = make_scorer(f1_score_not_binary, greater_is_better=True) if scoring == Consts.ScoreType.F1 else scoring.value

        random_state = Consts.listRandomStates[0]

        self.log("Creating a DECISION_TREE")
        clf = DecisionTreeClassifier()
        list_random_search.append(
            RandomizedSearchCV(
                estimator=clf,
                param_distributions=Consts.RandomGrid.decision_tree_grid,
                n_iter=n_iter,
                scoring=score,
                n_jobs=n_jobs,
                cv=cv,
                random_state=random_state
            )
        )

        self.log("Creating a RANDOM_FOREST")
        clf = RandomForestClassifier()
        list_random_search.append(
            RandomizedSearchCV(
                estimator=clf,
                param_distributions=Consts.RandomGrid.random_forest_grid,
                n_iter=n_iter,
                scoring=score,
                n_jobs=n_jobs,
                cv=cv,
                random_state=random_state
            )
        )

        return list_random_search

    # Utility function to report best scores
    @staticmethod
    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    def parameter_search_classifiers(self, classifier_types: Consts.ClassifierType=set(Consts.ClassifierType),
                                     scoring: Consts.ScoreType=Consts.ScoreType.ACCURACY) -> list:

        list_random_search: List[RandomizedSearchCV] = self.allocate_rand_search_classifiers(classifier_types, scoring)

        for random_search in list_random_search:
            start = time()
            random_search.fit(self.dict_dfs[Consts.FileSubNames.X_TRAIN], self.dict_dfs[Consts.FileSubNames.Y_TRAIN])
            end = time()

            #self.log(f"{random_search.estimator.__repr__()} ran for {end-start} while fitting")
            #self.report(random_search.cv_results_)

        return list_random_search

    def best_trained_model_by_validation(self, list_estimators: [RandomizedSearchCV]) -> (RandomizedSearchCV, float):

        list_model_score = [(model, model.score(self.dict_dfs[Consts.FileSubNames.X_VAL], self.dict_dfs[Consts.FileSubNames.Y_VAL])) for model in list_estimators]

        return max(list_model_score, key=lambda x: x[1])

    def inspect_validate_curve(self):
        pass

    def inspect_learning_curve(self, estimator):
        train_sizes, train_scores, valid_scores = learning_curve(
            estimator,
            self.dict_dfs[Consts.FileSubNames.X_TRAIN],
            self.dict_dfs[Consts.FileSubNames.Y_TRAIN]
        )


    def search_scoring_functions(self):

        for scoring_type in list(Consts.ScoreType):
            self.log("Scoring with {}".format(scoring_type.value))
            list_random_search = self.parameter_search_classifiers(scoring=scoring_type)
            model_score = self.best_trained_model_by_validation(list_random_search)
            self.log("estimator {} with score {}".format(model_score[0].estimator, model_score[1]))
            # self.inspect_learning_curve(model_score[0])

    def concatenate_train_and_val(self) -> (pd.DataFrame, pd.DataFrame):
        """
        :return: X_train + X_val, Y_train + Y_val
        """
        return np.concatenate((self.dict_dfs[Consts.FileSubNames.X_TRAIN], self.dict_dfs[Consts.FileSubNames.X_VAL]), axis=0), np.concatenate((self.dict_dfs[Consts.FileSubNames.Y_TRAIN], self.dict_dfs[Consts.FileSubNames.Y_VAL]), axis=0)

    def predict_the_winner(self, estimator) -> None:
        """
        save to a file!
        :param estimator:
        :return:
        """
        pass

    def predict_voters_distribution(self, estimator) -> None:
        """
        save to a file in Consts
        :param estimator:
        :return:
        """
        pass

    def predict_most_likely_voters(self, estimator) -> None:
        """
        We might change this to predict from estimators!
        save to a file in Consts
        :param estimator:
        :return:
        """
        pass

    def save_test_confusion_matrix(self, estimator) -> None:
        """
        save to a file in Consts.
        :param estimator:
        :return:
        """
        pass


def ex_3():

    use_the_same_model_for_all_tasks = True

    m = Modeling()

    # Use set 1
    set = 1

    # load the data from set 1.
    m.load_data(Consts.FileNames.FILTERED_AND_SCALED, set)

    # search for good parameters by using cross val.
    # TODO: default scoring is ACCURACY! is this what we want?
    list_random_search = m.parameter_search_classifiers()

    if use_the_same_model_for_all_tasks:

        best_estimator, best_estimator_score = m.best_trained_model_by_validation(list_random_search)

        m.predict_the_winner(best_estimator)
        m.predict_voters_distribution(best_estimator)
        m.predict_most_likely_voters(best_estimator)
        m.save_test_confusion_matrix(best_estimator)

def main():
    ex_3()


if __name__ == '__main__':
    main()