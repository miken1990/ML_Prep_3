import os
import sys
from time import time, strftime
from typing import List
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, learning_curve
from sklearn.tree import DecisionTreeClassifier

import Consts


# **********************************************************************************************************************#

class Modeling:
    dict_dfs_np = {d: None for d in list(Consts.FileSubNames)}
    dict_dfs_pd = {d: None for d in list(Consts.FileSubNames)}
    do_print = True
    logger = None

    def __init__(self, file_str=None, print_modeling: bool=False):
        self.do_print = print_modeling

        if self.do_print and file_str is not None:
            sys.stdout = open(file_str, 'w')

    def title(self, msg, decorator='*', decorator_len=80):
        if self.do_print:
            print(decorator * decorator_len)
            print('{}: {}'.format(strftime("%c"), msg))
            print(decorator * decorator_len)

    def log(self, msg):
        if self.do_print:
            print('{}: {}'.format(strftime("%c"), msg))

    def load_data(self, base: Consts.FileNames, set: int) -> None:
        """
        this method will load ready to use data for the training, validating, and testing sets.
        this implements stages 1, 3 and part of 6 in the assignment.
        :return:
        """
        self.title(f"Loading the data from {base}")
        # load train features and labels
        for d in list(Consts.FileSubNames):
            file_location = base.value.format(set, d.value)
            self.log(f"Loading {file_location}")
            if d in {Consts.FileSubNames.Y_TEST, Consts.FileSubNames.Y_VAL, Consts.FileSubNames.Y_TRAIN}:
                self.dict_dfs_np[d] = self._load_data(file_location)[Consts.VOTE_STR].as_matrix().ravel()
            else:
                self.dict_dfs_np[d] = self._load_data(file_location).as_matrix()
            self.dict_dfs_pd[d] = self._load_data(file_location)

    def _load_data(self, filePath):
        return read_csv(filePath, header=0, keep_default_na=True)

    def allocate_rand_search_classifiers(self, scoring: Consts.ScoreType) -> [RandomizedSearchCV]:
        list_random_search = []  # type: [RandomizedSearchCV]
        n_iter = 30
        n_jobs = 4
        cv = 10
        score = scoring.value

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
    def report(self, results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                self.log("Model with rank: {0}".format(i))
                self.log("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                self.log("Parameters: {0}".format(results['params'][candidate]))
                self.log("")

    def parameter_search_classifiers(self, scoring: Consts.ScoreType = Consts.ScoreType.ACCURACY) -> list:
        self.log(f"scoring with {scoring}")
        list_random_search: List[RandomizedSearchCV] = self.allocate_rand_search_classifiers(scoring)

        for random_search in list_random_search:
            random_search.fit(self.dict_dfs_np[Consts.FileSubNames.X_TRAIN],
                              self.dict_dfs_np[Consts.FileSubNames.Y_TRAIN])
            self.report(random_search.cv_results_)

        return list_random_search

    def best_trained_model_by_validation(self, list_estimators: [RandomizedSearchCV]) -> (RandomizedSearchCV, float):

        list_model_score = [(model, model.score(self.dict_dfs_np[Consts.FileSubNames.X_VAL],
                                                self.dict_dfs_np[Consts.FileSubNames.Y_VAL])) for model in
                            list_estimators]

        return max(list_model_score, key=lambda x: x[1])

    def search_scoring_functions(self):

        for scoring_type in list(Consts.ScoreType):
            self.log("Scoring with {}".format(scoring_type))
            list_random_search = self.parameter_search_classifiers(scoring=scoring_type)
            model_score = self.best_trained_model_by_validation(list_random_search)
            self.log("estimator {} with score {}".format(model_score[0].estimator, model_score[1]))

    def concatenate_train_and_val(self) -> (pd.DataFrame, pd.DataFrame):
        """
        :return: X_train + X_val, Y_train + Y_val
        """
        return np.concatenate(
            (self.dict_dfs_np[Consts.FileSubNames.X_TRAIN], self.dict_dfs_np[Consts.FileSubNames.X_VAL]),
            axis=0), np.concatenate(
            (self.dict_dfs_np[Consts.FileSubNames.Y_TRAIN], self.dict_dfs_np[Consts.FileSubNames.Y_VAL]), axis=0)

    def predict_the_winner(self, estimator, test_data, dir: Consts.EX3DirNames) -> None:
        """
        save to a file!
        :param estimator:
        :return: the name of the party with the majority of votes
        """

        y_pred = estimator.predict(test_data)
        y = y_pred.astype(np.int32)
        counts = np.bincount(y)
        winner = Consts.MAP_NUMERIC_TO_VOTE[np.argmax(counts)]
        file_path = dir.value + Consts.EX3FilNames.WINNER.value
        with open(file_path, "w") as file:
            file.write(winner)

        return y_pred

    def _predict_votes_aux(self, estimator, test_data):
        test_data_copy = test_data.copy()
        y_pred = estimator.predict(test_data_copy)
        test_data_copy[Consts.VOTE_STR] = pd.Series(y_pred)

        result = dict()
        for i in range(1, 12):
            result[i] = []

        for _, row in test_data_copy.iterrows():
            result[row[Consts.VOTE_STR]].append(row[Consts.INDEX_COL])

        return y_pred, result

    def predict_most_likely_voters(self, estimator, test_data, test_label, dir: Consts.EX3DirNames):
        y_pred, result = self._predict_votes_aux(estimator, test_data)

        # save predictions to file
        file_path = dir.value + Consts.EX3FilNames.MOST_LIKELY_PARTY.value
        with open(file_path, "w") as file:
            for i in range(1, 12):
                result[i] = [(int(item)) for item in result[i]]
                result[i].sort()
                string_to_write = Consts.MAP_NUMERIC_TO_VOTE[i] + f': {result[i]}'
                file.write(string_to_write + '\n')
        return y_pred, test_label

    def predict_voters_distribution(self, estimator, test_data, test_label, dir: Consts.EX3DirNames):
        """
        save to a file in Consts
        :param estimator:
        :return:
        """
        y_pred, result = self._predict_votes_aux(estimator, test_data)

        # save predictions to file
        file_path = dir.value + Consts.EX3FilNames.PREDICTED_DISTRIBUTION.value
        total_y = y_pred.shape[0]
        with open(file_path, "w") as file:
            for i in range(1, 12):
                string_to_write = Consts.MAP_NUMERIC_TO_VOTE[i] + f': {len(result[i]) / total_y}'
                file.write(string_to_write + '\n')
        return y_pred, test_label


    def print_test_confusion_matrix_and_test_error(self, y_pred, y_true) -> None:
        """
        save to a file in Consts.
        :return:
        """

        self.log("\n"+str(metrics.confusion_matrix(y_true[Consts.VOTE_STR], y_pred)))
        y_true_arr = np.array(y_true[Consts.VOTE_STR])
        list_equals = [1 if x == y_true_arr[index] else 0 for index, x in enumerate(y_pred)]
        self.log(np.average(list_equals))

    def plot_estimator_learning_curve(self, estimator, title=""):
        X, Y = self.concatenate_train_and_val()
        title = "Learning Curves " + title
        plot_learning_curve(estimator, title, X, Y, cv=6)
        plt.show()

    def draw_tree(self, tree):
        from sklearn.externals.six import StringIO
        from IPython.display import Image, display
        from sklearn.tree import export_graphviz
        import pydotplus
        self.log('Drawing Tree')
        with open("decision_tree.dot", "w") as f:
            f = export_graphviz(tree, out_file=f)


#**********************************************************************************************************************#

def create_files_ex3():
    for d in Consts.EX3DirNames:
        if not os.path.isdir(d.value):
            os.mkdir(d.value)


# **********************************************************************************************************************#

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


# **********************************************************************************************************************#

def ex_3(use_the_same_model_for_all_tasks: bool, use_multi_models_for_tasks: bool, show_learning_curves: bool,
         view_decision_tree: bool, print_ex3: bool) -> None:

    redirection_file = strftime("%y_%m_%d_%H_%M_%S") + ".txt"
    time_begining = datetime.datetime.now()
    print(time_begining.time())

    create_files_ex3()

    m = Modeling(file_str=redirection_file, print_modeling=print_ex3)
    m.title("Starting EX3")
    m.log("Time of start")
    # Use set 1
    set = 1

    # load the data from set 1.
    m.load_data(Consts.FileNames.FILTERED_AND_SCALED, set)

    if use_the_same_model_for_all_tasks:
        m.title("Same Estimator For All Tasks")
        list_random_search = m.parameter_search_classifiers()
        if view_decision_tree:
            decistion_tree = list_random_search[0].best_estimator_
            decistion_tree.fit(m.dict_dfs_np[Consts.FileSubNames.X_TRAIN], m.dict_dfs_np[Consts.FileSubNames.Y_TRAIN])
            # print(decistion_tree)
            m.draw_tree(decistion_tree)

        best_estimator, _ = m.best_trained_model_by_validation(list_random_search)

        if show_learning_curves:
            m.plot_estimator_learning_curve(best_estimator, "Single Estimator")

        m.predict_the_winner(best_estimator,
                             m.dict_dfs_np[Consts.FileSubNames.X_TEST],
                             Consts.EX3DirNames.SINGLE_ESTIMATOR)
        m.predict_voters_distribution(best_estimator,
                                      m.dict_dfs_pd[Consts.FileSubNames.X_TEST],
                                      m.dict_dfs_pd[Consts.FileSubNames.Y_TEST],
                                      Consts.EX3DirNames.SINGLE_ESTIMATOR)
        y_pred, y_true = m.predict_most_likely_voters(best_estimator,
                                                       m.dict_dfs_pd[Consts.FileSubNames.X_TEST],
                                                       m.dict_dfs_pd[Consts.FileSubNames.Y_TEST],
                                                       Consts.EX3DirNames.SINGLE_ESTIMATOR)

        m.title('Single Estimator Confusion Matrix')
        m.print_test_confusion_matrix_and_test_error(y_pred, y_true)
        # m.predict_most_likely_voters(best_estimator)
        # m.save_test_confusion_matrix(best_estimator)
    if use_multi_models_for_tasks:
        m.title("Creating an estimator for each task")
        m.log("Training an estimator for the winner")
        list_random_search_winner =  m.parameter_search_classifiers(scoring=Consts.ScoreType.WINNER_PRECISION)
        m.log("Training an estimator for the distribution")
        list_random_search_distribution =  m.parameter_search_classifiers(scoring=Consts.ScoreType.DISTRIBUTION)
        m.log("Training an estimator for the accuracy")
        list_random_search_accuracy =  m.parameter_search_classifiers(scoring=Consts.ScoreType.ACCURACY)

        m.log("Getting the best estimator per task")
        winner_estimator, _ = m.best_trained_model_by_validation(list_random_search_winner)
        distribution_estimator, _ = m.best_trained_model_by_validation(list_random_search_distribution)
        accuracy_estimator, _ = m.best_trained_model_by_validation(list_random_search_accuracy)
        if show_learning_curves:
            m.plot_estimator_learning_curve(winner_estimator, "Winner Estimator")
            m.plot_estimator_learning_curve(distribution_estimator, "Distribution Estimator")
            m.plot_estimator_learning_curve(accuracy_estimator, "Accuracy Estimator")
        y_true = m.dict_dfs_pd[Consts.FileSubNames.Y_TEST]
        m.title("Predicting the winning party")
        y_pred = m.predict_the_winner(winner_estimator,
                                      m.dict_dfs_np[Consts.FileSubNames.X_TEST],
                                      Consts.EX3DirNames.MULTI_ESTIMATORS)
        m.log("Confusion Matrix for the Winner estimator")
        m.print_test_confusion_matrix_and_test_error(y_pred, y_true)
        m.title("Predicting the distribution")
        y_pred, _ = m.predict_voters_distribution(distribution_estimator,
                                               m.dict_dfs_pd[Consts.FileSubNames.X_TEST],
                                               m.dict_dfs_pd[Consts.FileSubNames.Y_TEST],
                                               Consts.EX3DirNames.SINGLE_ESTIMATOR)
        m.log("Confusion Matrix for the distribution estimator")
        m.print_test_confusion_matrix_and_test_error(y_pred, y_true)
        m.title("Predicting the Most Likely")
        y_pred, _ = m.predict_most_likely_voters(accuracy_estimator,
                                                      m.dict_dfs_pd[Consts.FileSubNames.X_TEST],
                                                      m.dict_dfs_pd[Consts.FileSubNames.Y_TEST],
                                                      Consts.EX3DirNames.SINGLE_ESTIMATOR)
        m.log("Confusion Matrix for the Accuracy estimator")
        m.print_test_confusion_matrix_and_test_error(y_pred, y_true)

# **********************************************************************************************************************#
