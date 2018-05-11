import numpy as np
import Consts
import pandas as pd
import os
from pandas import read_csv, to_numeric, Series
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt



class ElectionsDataPreperation:
    """ main class that is used for data preperation
    """

    def __init__(self, sInputFileTrain, sInputFileVal, sInputFileTest, sInputFileTrainLabel, sInputFileValLabel,
                 sInputFileTestLabel):

        # fields that contain path to X files of train, val and test
        self.sInputFileTrain = sInputFileTrain
        self.sInputFileVal = sInputFileVal
        self.sInputFileTest = sInputFileTest

        # fields that contain path to y files of train, val, test
        self.sInputFileTrainLabel = sInputFileTrainLabel
        self.sInputFileValLabel = sInputFileValLabel
        self.sInputFileTestLabel = sInputFileTestLabel

        # fields that contain x and Y for train, test and validation
        self.trainData = None
        self.trainLabels = None
        self.valData = None
        self.valLabels = None
        self.testData = None
        self.testLabels = None

    def loadAndImpute(self, lDataTypes=None):
        """ lDataTypes is a list with following values ['test', 'validation'], the list determines if the data load,
        renaming and imputation will happen on the test and validation sets
        """
        self.loadData(lDataTypes)
        self._changeStringToValues(lDataTypes)
        self._changeVoteToNumber(lDataTypes)
        # first we impute train
        self._dataImpute(self.trainData, self.trainData, self.sInputFileTrain)

        if ("test" in lDataTypes):
            self._dataImpute(self.trainData, self.testData, self.sInputFileTest)

        if ("validation" in lDataTypes):
            self._dataImpute(self.trainData, self.valData, self.sInputFileVal)


    def loadData(self, lDataTypes = []):
        """lDataTypes is a list with following values ['test', 'validation'], the list determines if the data load,
        renaming and imputation will happen on the test and validation sets
        """
        trainFileNameX = self.sInputFileTrain + '.csv'
        self.trainData = read_csv(trainFileNameX, header=0, keep_default_na=True)
        trainFileNameY = self.sInputFileTrainLabel + '.csv'
        self.trainLabels = read_csv(trainFileNameY, header=0)

        if ('test' in lDataTypes):
            testFileNameX = self.sInputFileTest + '.csv'
            self.testData = read_csv(testFileNameX, header=0, keep_default_na=True)
            testFileNameY = self.sInputFileTestLabel + '.csv'
            self.testLabels = read_csv(testFileNameY, header=0)

        if ('validation' in lDataTypes):
            valFileNameX = self.sInputFileVal + '.csv'
            self.valData = read_csv(valFileNameX, header=0, keep_default_na=True)
            valFileNameY = self.sInputFileValLabel + '.csv'
            self.valLabels = read_csv(valFileNameY, header=0)


    def filterFeatures(self, lDataTypes = []):
        """filters unnecessary features from the dataset
        """
        listOrderedSelectedFeatures = [feature for feature in self.trainData.columns if feature in Consts.setSelectedFeatures]
        self.trainData = self.trainData[listOrderedSelectedFeatures]
        if 'validation' in lDataTypes:
            self.valData = self.valData[listOrderedSelectedFeatures]
        if 'test' in lDataTypes:
            self.testData = self.testData[listOrderedSelectedFeatures]


    def _changeStringToValues(self, lDataTypes):
        """lDataTypes is a list with following values['test', 'validation'], the list determines if the
        _changeStringToValues will happen on the test and validation sets.
        After the change, the files are saved with Numeric suffix
        """
        self._changeStringToValuesAux(self.trainData, self.sInputFileTrain)
        self._changeVoteToNumber(lDataTypes)
        # self._fillBoolValues(self.trainData)
        # self._fillTrioValues(self.trainData)
        # self._fillHotSpot(self.trainData, Consts.listSymbolicColumns)
        # # remove previous columns containing strings
        # self.trainData = self.trainData.drop(Consts.listNonNumeric, axis=1)
        # self.trainData = self.trainData.drop(self.trainData.columns[0], axis=1)
        # ElectionsDataPreperation.fixNegativeVals(self.trainData)
        # trainPath = self.sInputFileTrain + 'Numeric.csv'
        # self.trainData.to_csv(trainPath)

        if ('test' in lDataTypes):
            self._changeStringToValuesAux(self.testData, self.sInputFileTest)

        if ('validation' in lDataTypes):
            self._changeStringToValuesAux(self.valData, self.sInputFileVal)

    def _changeStringToValuesAux(self, data, sInputFilePath):
        self._fillBoolValues(data)
        self._fillTrioValues(data)
        self._fillHotSpot(data, Consts.listSymbolicColumns)
        # remove previous columns containing strings
        data = data.drop(Consts.listNonNumeric, axis=1)
        # data = data.drop(data.columns[0], axis=1)
        ElectionsDataPreperation.fixNegativeVals(data)
        Path = sInputFilePath + 'Numeric.csv'
        data.to_csv(Path)

    def _changeVoteToNumber(self, lDataTypes=None):
        """lDataTypes is a list with following values['test', 'validation'], the list determines if the
        _changeVoteToNumber will happen on the test and validation sets.
        """
        self.trainLabels['Vote'] = self.trainLabels['Vote'].map({'Greens': 10, 'Pinks': 9, 'Purples': 8,
                                                                                 'Blues': 7, 'Whites': 6, 'Browns': 5,
                                                                                 'Yellows': 4, 'Reds': 3,
                                                                                 'Turquoises': 2,
                                                                                 'Greys': 1, 'Oranges': 11})
        self.trainLabels = self.trainLabels[["Vote"]]
        self.trainLabels.to_csv(self.sInputFileTrainLabel + 'Numeric.csv')

        if ('validation' in lDataTypes):
            self.valLabels['Vote'] = self.valLabels['Vote'].map({'Greens': 10, 'Pinks': 9, 'Purples': 8,
                                                                                 'Blues': 7, 'Whites': 6, 'Browns': 5,
                                                                                 'Yellows': 4, 'Reds': 3,
                                                                                 'Turquoises': 2,
                                                                                 'Greys': 1, 'Oranges': 11})
            self.valLabels = self.valLabels[["Vote"]]
            self.valLabels.to_csv(self.sInputFileValLabel + 'Numeric.csv')
        if ('test' in lDataTypes):
            self.testLabels['Vote'] = self.testLabels['Vote'].map({'Greens': 10, 'Pinks': 9, 'Purples': 8,
                                                                                 'Blues': 7, 'Whites': 6, 'Browns': 5,
                                                                                 'Yellows': 4, 'Reds': 3,
                                                                                 'Turquoises': 2,
                                                                                 'Greys': 1, 'Oranges': 11})
            self.testLabels = self.testLabels[["Vote"]]
            self.testLabels.to_csv(self.sInputFileTestLabel + 'Numeric.csv')

    def _dataImpute(self, trainData, imputeData, sFileName):
        data_with_NaN = imputeData.isnull().any(axis=1)
        data_with_NaN = np.where(data_with_NaN)
        self.closestFitNanFill(trainData, imputeData, data_with_NaN, sFileName)

    def closestFitNanFill(self, trainData, imputeData, data_with_nan, sFileName):
        """ finds the closest row to each row that contains NaN and fills the NaNs accordingly
        output is saved in a file - "self.sInputFile"
        """
        # fill the dict with r value and isNumeric boolean for closest fit function
        dist_args_dict = dict()
        for feature in trainData.keys():
            isNumeric = feature in Consts.setNumericFeatures
            d = to_numeric(trainData[feature], errors='coerce')
            max_val = d.max(axis=0)
            min_val = d.min(axis=0)
            dist_args_dict[feature] = (max_val - min_val, isNumeric)

        inf = Consts.inf
        trainDataArray = trainData.as_matrix()
        imputeDataArray = imputeData.as_matrix()
        source_null_indexes = None
        dest_null_indexes = None
        nearestRowDict = dict()

        for index in data_with_nan[0]:
            source = imputeDataArray[index]
            source_null_indexes = np.argwhere(np.isnan(source)).transpose()[0]
            nearest_row_rating = inf
            for destIndex in range(trainDataArray.shape[0]):
                destination = trainDataArray[destIndex]
                dest_null_indexes = np.argwhere(np.isnan(destination)).transpose()[0]
                if np.intersect1d(source_null_indexes, dest_null_indexes, assume_unique=True).size:
                    continue
                dist = self._distRow(source, destination, dist_args_dict)
                # update index for each row with NaN, with nearest neighbour
                if dist < nearest_row_rating:
                    nearest_row_rating = dist
                    nearestRowDict[index] = destIndex

            destination = trainDataArray[nearestRowDict[index]]
            source[source_null_indexes] = destination[source_null_indexes]
            imputeDataArray[index] = source
            # print(source)
            print(index.__repr__() + ' ' + nearestRowDict[index].__repr__())
        # print(sFileName)
        imputeData = pd.DataFrame(imputeDataArray, columns=imputeData.columns.values)
        imputeData = imputeData.loc[:, ~imputeData.columns.str.contains('^Unnamed')]
        imputeData.to_csv(sFileName + 'No_Nan.csv')

    def _fillBoolValues(self, data):
        """ replaces Bool columns with 1, 0 and NaN
        """
        # no boolean features at all
        pass
        # data['Looking_at_poles_int'] = data['Looking_at_poles_results'].map({'Yes': 1, 'No': -1, 'NA': np.nan})
        # data['Married_int'] = data['Married'].map({'Yes': 1, 'No': -1, 'NA': np.nan})
        # data['Gender_int'] = data['Gender'].map({'Male': 1, 'Female': -1, 'NA': np.nan})
        # data['Voting_time_int'] = data['Voting_Time'].map({'After_16:00': 1, 'By_16:00': -1, 'NA': np.nan})
        # data['Financial_agenda_matters_int'] = data['Financial_agenda_matters'].map({'Yes': 1, 'No': -1, 'NA': np.nan})

    def _fillTrioValues(self, data):
        data['Will_vote_only_large_party_int'] = data['Will_vote_only_large_party'].map(
            {'Yes': 1, 'Maybe': 0, 'No': -1, 'NA': np.nan})
        # data['Age_group_int'] = data['Age_group'].map(
        #     {'Below_30': -1, '30-45': 0, '45_and_up': 1, 'NA': np.nan})

    def _fillHotSpot(self, data, featureList):
        for feature in featureList:
            lFeatures = data[feature].unique()
            for category in lFeatures:
                if 'nan' == category.__repr__():
                    continue
                dMapping = {x: 1 if x == category else 0 for x in lFeatures}
                dMapping['NA'] = np.nan
                data[feature + '_' + category] = data[feature].map(dMapping)

    def _distFeature(self, xi, yi, r, isContinues):
        """computed distance between features
        """
        if np.isnan(xi) or np.isnan(yi):
            return 1
        if xi == yi:
            return 0
        if isContinues:
            return abs(xi - yi) / r
        return 1

    def _distRow(self, source, destination, dist_args_dict):
        """computes distance between rows
        """
        res = 0
        for xi, yi, isNum_r_tuple in zip(source, destination, dist_args_dict.values()):
            res += self._distFeature(xi, yi,isNum_r_tuple[0], isNum_r_tuple[1])
        return res

    def removeAbove95Corr(self):
        corr_matrix = self.trainData.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        upper.to_csv(self.sInputFileTest+ 'CorrMatrix.csv')
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        self.trainData = self.trainData.drop(to_drop, axis=1)

    def sequential_baskward_selection(df: pd.DataFrame, J: callable) -> dict:
        base = {feature for feature in df.keys()}
        result = {len(base): base}

        for i in reversed(range(1, len(base))):
            target_feature = max(
                [(new_feature, J(result[i + 1].difference({new_feature})))
                 for new_feature in result[i + 1]],
                key=lambda x: x[1])[0]
            result[i] = result[i + 1].difference({target_feature})

        return result

    @staticmethod
    def fixNegativeVals(df: pd.DataFrame) -> None:
        df[Consts.listFixNegateVals] = np.absolute(df[Consts.listFixNegateVals])

# end class ElectionsDataPreparation

class DataSplit:
    """ receives File path, loads data and splits the data in a stratified way
    """
    def __init__(self, sFilePath):
        self.dataset = read_csv(sFilePath, header=0, keep_default_na=True)
        self.data = self.dataset.loc[:, self.dataset.columns != 'Vote']
        self.labels = self.dataset.loc[:, self.dataset.columns =='Vote']

    def saveDataSetsToCsv(self):
        """ save splitted data to csv
        """
        tDataSets = self.stratifySplit()

        if not os.path.isdir('datasets'):
            os.mkdir('datasets')

        if not os.path.isdir('datasets/1'):
            os.mkdir('datasets/1')

        if not os.path.isdir('datasets/2'):
            os.mkdir('datasets/2')


        for i, dataSet in enumerate(tDataSets):
            dataSet[0].to_csv('datasets/' + str(i + 1) + '/X_train' + '{}'.format(i+1) + '.csv')
            dataSet[1].to_csv('datasets/' + str(i + 1) + '/X_val' + '{}'.format(i + 1) + '.csv')
            dataSet[2].to_csv('datasets/' + str(i + 1) + '/X_test' + '{}'.format(i+1) + '.csv')
            dataSet[3].to_csv('datasets/' + str(i + 1) + '/Y_train' + '{}'.format(i+1) + '.csv')
            dataSet[4].to_csv('datasets/' + str(i + 1) + '/Y_val' + '{}'.format(i + 1) + '.csv')
            dataSet[5].to_csv('datasets/' + str(i + 1) + '/Y_test' + '{}'.format(i+1) + '.csv')

    def stratifySplit(self):
        """splits the data into three different data sets
        """
        X_train_second, X_test_second, y_train_second, y_test_second = train_test_split(self.data, self.labels,
                                                                                        train_size=0.85,
                                                                                        shuffle=True,
                                                                                        random_state=376674226)


        X_train_second, X_val_second, y_train_second, y_val_second = train_test_split(X_train_second,
                                                                                      y_train_second,
                                                                                      train_size=0.8235,
                                                                                      shuffle=True,
                                                                                      random_state=493026216,
                                                                                      stratify=y_train_second)

        X_train_third, X_test_third, y_train_third, y_test_third = train_test_split(self.data, self.labels,
                                                                                    train_size=0.85, shuffle=True,
                                                                                    random_state=404629562,
                                                                                    stratify=self.labels)

        X_train_third, X_val_third, y_train_third, y_val_third = train_test_split(X_train_third, y_train_third,
                                                                                  train_size=0.8235, shuffle=True,
                                                                                  random_state=881225405,
                                                                                  stratify=y_train_third)

        return [(X_train_second, X_val_second, X_test_second, y_train_second, y_val_second, y_test_second),
                (X_train_third, X_val_third, X_test_third, y_train_third, y_val_third, y_test_third)]

# end method DataSplit


if __name__ == '__main__':

    firstSetPrep = ElectionsDataPreperation('datasets/1/X_train1No_Nan', 'datasets/1/X_val1',
                                            'datasets/1/X_test1Numeric - Copy', 'datasets/1/Y_train1',
                                            'datasets/1/Y_val1', 'datasets/1/Y_test1')

    firstSetPrep.loadData(Consts.listAdditionalDataPreparation)
    firstSetPrep._dataImpute(firstSetPrep.trainData, firstSetPrep.testData, 'datasets/1/X_test1-copy')









