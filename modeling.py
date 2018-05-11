from enum import Enum


class Modeling:
    train_set = None
    validation_set = None
    test_set = None

    class ClassifierType(Enum):
        DECISION_TREE = 1

    class ScoreType(Enum):
        ACCURACY = 1
        ERROR = 2
        PRECISION = 3
        RECALL = 4
        FP_RATE = 5
        F1_SCORE = 6 # Harmonic 2/(1/precision + 1/recall)

    def load_data(self, file_path: str) -> None:
        """
        this method will load ready to use data for the training, validating, and testing sets.
        this implements stages 1, 3 and part of 6 in the assignment.
        :param file_path: the location of the data csv.
        :return:
        """
        pass

    def create_and_train_classifiers(self, classifier_type_list: [ClassifierType]) -> list:
        """
        this method will train a few classifiers and store them in classifiers list.
        default amount is 2, as required in the assignment.
        :param amount: the amount of wanted classifiers
        :return: a list of classifiers
        """
    def create_and_train_classifier_by_train_and_validation(self, classifier_type: ClassifierType):
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

