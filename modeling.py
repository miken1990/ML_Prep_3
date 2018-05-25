import os
from time import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, learning_curve
from sklearn.tree import DecisionTreeClassifier

import Consts

"""
TODO list:

    1. function receives an estimator and predicts a list of most likely to vote for each party
        a. from fresh test set.
        b. save each list to a separate csv. (file names in new directory in Consts!)
    2. compare scoring functions (in a function) and save outputs to csv ot print. save the wanted scoring method.
        - search the web for their plots.
        - explain why they are good for us.
    3. Identify factors which by manipulating we can change the outcome of the elections

    4. use different models for each of the prediction tasks
    if there is time:
"""
#**********************************************************************************************************************#

# Scoring functions:

def f1_score_not_binary(y_true, y_pred):
    return np.mean(f1_score(y_true, y_pred, average=None))

#**********************************************************************************************************************#

class Modeling:
    dict_dfs_np = {d: None for d in list(Consts.FileSubNames)}
    do_print = True

    def __init__(self):
        self.dict_dfs_np = {d: None for d in list(Consts.FileSubNames)}
        self.dict_dfs_pd = {d: None for d in list(Consts.FileSubNames)}

    def log(self, msg):
        if self.do_print:
            print(msg)

    def load_data(self, base: Consts.FileNames, set: int) -> None:
        """
        this method will load ready to use data for the training, validating, and testing sets.
        this implements stages 1, 3 and part of 6 in the assignment.
        :return:
        """
        self.log(f"Loading the data from {base}")
        # load train features and labels
        for d in list(Consts.FileSubNames):
            file_location = base.value.format(set, d.value)
            if d in {Consts.FileSubNames.Y_TEST, Consts.FileSubNames.Y_VAL, Consts.FileSubNames.Y_TRAIN }:
                self.dict_dfs_np[d] = self._load_data(file_location)[Consts.VOTE_STR].as_matrix().ravel()
            else:
                self.dict_dfs_np[d] = self._load_data(file_location).as_matrix()
            self.dict_dfs_pd[d] = self._load_data(file_location)

    def _load_data(self, filePath):
        self.log(f"Loading {filePath}")
        return read_csv(filePath, header=0, keep_default_na=True)

    def allocate_rand_search_classifiers(self, classifier_types: {Consts.ClassifierType},
                                         scoring: Consts.ScoreType) -> [RandomizedSearchCV]:
        list_random_search = []  # type: [RandomizedSearchCV]
        n_iter = 1 #20
        n_jobs = 4
        cv = 4
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
            random_search.fit(self.dict_dfs_np[Consts.FileSubNames.X_TRAIN], self.dict_dfs_np[Consts.FileSubNames.Y_TRAIN])
            end = time()

            #self.log(f"{random_search.estimator.__repr__()} ran for {end-start} while fitting")
            #self.report(random_search.cv_results_)

        return list_random_search

    def best_trained_model_by_validation(self, list_estimators: [RandomizedSearchCV]) -> (RandomizedSearchCV, float):

        list_model_score = [(model, model.score(self.dict_dfs_np[Consts.FileSubNames.X_VAL], self.dict_dfs_np[Consts.FileSubNames.Y_VAL])) for model in list_estimators]

        return max(list_model_score, key=lambda x: x[1])

    def inspect_validate_curve(self):
        pass

    def inspect_learning_curve(self, estimator):
        train_sizes, train_scores, valid_scores = learning_curve(
            estimator,
            self.dict_dfs_np[Consts.FileSubNames.X_TRAIN],
            self.dict_dfs_np[Consts.FileSubNames.Y_TRAIN]
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
        return np.concatenate((self.dict_dfs_np[Consts.FileSubNames.X_TRAIN], self.dict_dfs_np[Consts.FileSubNames.X_VAL]), axis=0), np.concatenate((self.dict_dfs_np[Consts.FileSubNames.Y_TRAIN], self.dict_dfs_np[Consts.FileSubNames.Y_VAL]), axis=0)

    def predict_the_winner(self, estimator, test_data, test_label) -> None:
        """
        save to a file!
        :param estimator:
        :return: the name of the party with the majority of votes
        """
        y_pred = estimator.predict(test_data)
        counts = np.bincount(y_pred)
        winner = Consts.MAP_NUMERIC_TO_VOTE[np.argmax(counts)]
        file_path = Consts.EX3DirNames.SINGLE_ESTIMATOR.value + Consts.EX3FilNames.WINNER.value
        with open(file_path, "w") as file:
            file.write(winner)


    def predict_voters_distribution(self, estimator, test_data, test_label) -> None:
        """
        save to a file in Consts
        :param estimator:
        :return:
        """
        y_pred = estimator.predict(test_data)
        test_data[Consts.VOTE_STR] = pd.Series(y_pred)
        result = dict()
        for i in range(1, 12):
            result[i] = []

        for _, row in test_data.iterrows():
            result[row[Consts.VOTE_STR]].append(row[Consts.INDEX_COL])

        # save predictions to file
        file_path = Consts.EX3DirNames.SINGLE_ESTIMATOR.value + Consts.EX3FilNames.PREDICTED_DISTRIBUTION.value
        with open(file_path, "w") as file:
            for i in range(1, 12):
                result[i] = [(int(item)) for item in result[i]]
                result[i].sort()
                string_to_write = Consts.MAP_NUMERIC_TO_VOTE[i] + f': {result[i]}'
                file.write(string_to_write + '\n')

        # save confusion matrix
        self.save_test_confusion_matrix(y_pred, test_label)


    def predict_most_likely_voters(self, estimator) -> None:
        """
        We might change this to predict from estimators!
        save to a file in Consts
        :param estimator:
        :return:
        """
        pass

    def save_test_confusion_matrix(self, y_pred, y_true) -> None:
        """
        save to a file in Consts.
        :param estimator:
        :return:
        """

        print(metrics.confusion_matrix(y_true[Consts.VOTE_STR], y_pred))

    def plot_estimator_learning_curve(self, estimator):
        X, Y = self.concatenate_train_and_val()

        title = "Learning Curves"

        plot_learning_curve(estimator, title, X, Y, cv=6)

        plt.show()

#**********************************************************************************************************************#

def create_files_ex3():
    for d in Consts.EX3DirNames:
        if not os.path.isdir(d.value):
            os.mkdir(d.value)

#**********************************************************************************************************************#

# http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

#**********************************************************************************************************************#

def ex_3():

    use_the_same_model_for_all_tasks = True
    show_learning_curves = False

    create_files_ex3()

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

        if show_learning_curves:
            m.plot_estimator_learning_curve(best_estimator)

        m.predict_the_winner(best_estimator, m.dict_dfs_np[Consts.FileSubNames.X_TEST],
                             m.dict_dfs_np[Consts.FileSubNames.Y_TEST])
        m.predict_voters_distribution(best_estimator, m.dict_dfs_pd[Consts.FileSubNames.X_TEST],
                             m.dict_dfs_pd[Consts.FileSubNames.Y_TEST])
        # m.predict_most_likely_voters(best_estimator)
        # m.save_test_confusion_matrix(best_estimator)

#**********************************************************************************************************************#

def main():
    ex_3()

if __name__ == '__main__':
    main()