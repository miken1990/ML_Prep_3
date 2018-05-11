import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn import metrics


def sequential_forward_selection(clf, X: pd.DataFrame, y: pd.DataFrame, k) -> list:
    """
    calculate for each available amount of features the best set.
    like in the tutor, large sets contain the smaller sets.

    :return: a dict indexed by int's, each entry contains a set of the best features selected for this entry.
    """


    X = X.loc[:, X.columns != 'Unnamed: 0']
    base = [feature for feature in X.keys()]
    bestIndexes = dict()
    bestScores = dict()
    X = X.as_matrix()
    y = y.as_matrix().ravel()

    for i in range(k):
        bestScore = 0
        for j in range(0, len(base)):
            if j in bestIndexes.values():
                continue
            currIndexes = [bestIndexes[l] for l in range(i)]
            currIndexes.append(j)
            currX = X[:, currIndexes]
            tempScore = metrics.accuracy_score(y, cross_val_predict(clf, currX, y, cv=3))
            if tempScore > bestScore:
                bestScore = tempScore
                bestIndexes[i] = j
                bestScores[i] = bestScore

    indexByOrder = []
    bestFeatures = []
    print(bestScores)
    for l in bestIndexes.keys():
        indexByOrder.append(bestIndexes[l])
        bestFeatures.append(base[bestIndexes[l]])

    return bestIndexes, bestFeatures, bestScores

def sfsAux(clf, X: pd.DataFrame, y: pd.DataFrame, k):
    bestIndexes, bestFeatures, bestScores = sequential_forward_selection(clf, X, y, k)
    maxScoreIndex = max(bestScores.items(), key=lambda x: x[1])[0]
    return [bestFeatures[x] for x in range(maxScoreIndex + 1)]


