import os
from enum import Enum

import pandas as pd

import Consts
import relief
from ElectionsDataPreperation import ElectionsDataPreperation as EDP, DataSplit
from scale_data import ScaleData
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sfs import sfsAux


class Stages:
    # Stages:
    do_print = True
    do_get_raw_data = True
    do_filter_features = True
    do_load_and_impute = True
    do_scale = True
    do_scale_load_file = False
    do_feature_selection = False
    do_feature_selection_load_data = False
    do_removeAbove95Corr = False
    do_sfs = False
    do_relief = False
    get_correlations = False


amount_of_sets = 1


def create_files():
    for d in Consts.DirNames:
        if d == Consts.DirNames.DATA_SETS:
            if not os.path.isdir(d.value):
                os.mkdir(d.value)

        else:
            for i in range(1, 3):
                if not os.path.isdir(d.value.format(i)):
                    os.mkdir(d.value.format(i))


def main():
    create_files()

    # FIRST STEP: Get the data and split it in to 2 groups of 3 data sets.
    # we need to bring the initial file only once. while working on it, it is rather efficient to work on local files
    # yet we'd like to be able to get the files and fall threw these steps again if needed.
    if Stages.do_get_raw_data:
        if Stages.do_print:
            print("Stage 1: Importing the data")
        ds = DataSplit(Consts.FileNames.RAW_FILE_PATH.value)
        ds.saveDataSetsToCsv()

    # SECOND STEP: Prepare the data for work.
    secondStepPrep_dict = dict()
    scaleData_dict = dict()
    if Stages.do_load_and_impute:

        if Stages.do_print:
            print("Stage 2: Fixing nan and outliers")

        for i in range(1, amount_of_sets + 1):
            # start the preparing data class
            secondStepPrep_dict[i] = EDP(Consts.FileNames.RAW_AND_SPLITED.value.format(i, "X_train", i),
                                         Consts.FileNames.RAW_AND_SPLITED.value.format(i, "X_val", i),
                                         Consts.FileNames.RAW_AND_SPLITED.value.format(i, "X_test", i),
                                         Consts.FileNames.RAW_AND_SPLITED.value.format(i, "Y_train", i),
                                         Consts.FileNames.RAW_AND_SPLITED.value.format(i, "Y_val", i),
                                         Consts.FileNames.RAW_AND_SPLITED.value.format(i, "Y_test", i))
            # Load the data from csv.
            # Swap strings to numeric values
            # Impute missing data
            # Impute outlier and typos
            secondStepPrep_dict[i].loadData(Consts.listAdditionalDataPreparation)
            if Stages.do_filter_features:
                secondStepPrep_dict[i].filterFeatures(Consts.listAdditionalDataPreparation)
                secondStepPrep_dict[i].trainData.to_csv(
                    Consts.FileNames.RAW_AND_FILTERED.value.format(i, Consts.FileSubNames.X_TRAIN.value))
                secondStepPrep_dict[i].valData.to_csv(
                    Consts.FileNames.RAW_AND_FILTERED.value.format(i, Consts.FileSubNames.X_VAL.value))
                secondStepPrep_dict[i].testData.to_csv(
                    Consts.FileNames.RAW_AND_FILTERED.value.format(i, Consts.FileSubNames.X_TEST.value))

            secondStepPrep_dict[i]._changeStringToValues(Consts.listAdditionalDataPreparation)

            secondStepPrep_dict[i] = EDP(
                Consts.FileNames.FILTERED_AND_NUMERIC_NAN.value.format(i, Consts.FileSubNames.X_TRAIN.value),
                Consts.FileNames.FILTERED_AND_NUMERIC_NAN.value.format(i, Consts.FileSubNames.X_VAL.value),
                Consts.FileNames.FILTERED_AND_NUMERIC_NAN.value.format(i, Consts.FileSubNames.X_TEST.value),
                Consts.FileNames.FILTERED_AND_NUMERIC_NAN.value.format(i, Consts.FileSubNames.Y_TRAIN.value),
                Consts.FileNames.FILTERED_AND_NUMERIC_NAN.value.format(i, Consts.FileSubNames.Y_VAL.value),
                Consts.FileNames.FILTERED_AND_NUMERIC_NAN.value.format(i, Consts.FileSubNames.Y_TEST.value)
            )

            secondStepPrep_dict[i].loadData(Consts.listAdditionalDataPreparation)

            # secondStepPrep_dict[i]._changeStringToValues(Consts.listAdditionalDataPreparation)

            secondStepPrep_dict[i]._dataImpute(secondStepPrep_dict[i].trainData, secondStepPrep_dict[i].trainData,
                                               secondStepPrep_dict[i].sInputFileTrain)

            secondStepPrep_dict[i]._dataImpute(secondStepPrep_dict[i].trainData, secondStepPrep_dict[i].valData,
                                               secondStepPrep_dict[i].sInputFileVal)

            secondStepPrep_dict[i]._dataImpute(secondStepPrep_dict[i].trainData, secondStepPrep_dict[i].testData,
                                               secondStepPrep_dict[i].sInputFileTest)

    if Stages.do_scale:
        if Stages.do_print:
            print("Stage 3: Scale the data")
        for i in range(1, amount_of_sets + 1):
            # start the preparing data class
            if Stages.do_scale_load_file:
                secondStepPrep_dict[i] = EDP(
                    Consts.FileNames.FILTERED_AND_NUMERIC_NONAN.value.format(i, Consts.FileSubNames.X_TRAIN.value),
                    Consts.FileNames.FILTERED_AND_NUMERIC_NONAN.value.format(i, Consts.FileSubNames.X_VAL.value),
                    Consts.FileNames.FILTERED_AND_NUMERIC_NONAN.value.format(i, Consts.FileSubNames.X_TEST.value),
                    Consts.FileNames.FILTERED_AND_NUMERIC_NONAN.value.format(i, Consts.FileSubNames.Y_TRAIN.value),
                    Consts.FileNames.FILTERED_AND_NUMERIC_NONAN.value.format(i, Consts.FileSubNames.Y_VAL.value),
                    Consts.FileNames.FILTERED_AND_NUMERIC_NONAN.value.format(i, Consts.FileSubNames.Y_TEST.value)
                )
                secondStepPrep_dict[i].loadData(Consts.listAdditionalDataPreparation)

            initial_corr = secondStepPrep_dict[i].trainData.corr()
            if Stages.get_correlations:
                initial_corr.to_csv(Consts.FileNames.SUMMARY.value.format(i, 'initial_corr'))

            # scale the data
            scaleData_dict[i] = ScaleData()  # type: ScaleData
            scaleData_dict[i].scale_train(secondStepPrep_dict[i].trainData)
            scaleData_dict[i].scale_test(secondStepPrep_dict[i].valData)
            scaleData_dict[i].scale_test(secondStepPrep_dict[i].testData)
            # scaleData_dict[i].scale_test(secondStepPrep_dict[i].testData)
            secondStepPrep_dict[i].trainData.to_csv(
                Consts.FileNames.FILTERED_AND_SCALED.value.format(i, Consts.FileSubNames.X_TRAIN.value)
            )
            secondStepPrep_dict[i].valData.to_csv(
                Consts.FileNames.FILTERED_AND_SCALED.value.format(i, Consts.FileSubNames.X_VAL.value)
            )
            secondStepPrep_dict[i].testData.to_csv(
                Consts.FileNames.FILTERED_AND_SCALED.value.format(i, Consts.FileSubNames.X_TEST.value)
            )

            second_corr = secondStepPrep_dict[i].trainData.corr()
            if Stages.get_correlations:
                second_corr.to_csv(Consts.FileNames.SUMMARY.value.format(i,'Scaled_corr_diff'))
                (second_corr - initial_corr).abs().to_csv(Consts.FileNames.SUMMARY.value.format(i, 'Scaled_corr_diff'))

    if Stages.do_feature_selection:
        if Stages.do_print:
            print("Stage 4: Selecting relevant features")
        # relief + sfs + correlation matrix
        for i in range(1, amount_of_sets + 1):
            # load the data from the previous stage
            if Stages.do_feature_selection_load_data:
                secondStepPrep_dict[i] = EDP(
                    Consts.FileNames.FILTERED_AND_SCALED.value.format(i, Consts.FileSubNames.X_TRAIN.value),
                    Consts.FileNames.FILTERED_AND_SCALED.value.format(i, Consts.FileSubNames.X_VAL.value),
                    Consts.FileNames.FILTERED_AND_SCALED.value.format(i, Consts.FileSubNames.X_TEST.value),
                    Consts.FileNames.FILTERED_AND_NUMERIC_NAN.value.format(i, Consts.FileSubNames.Y_TRAIN.value),
                    Consts.FileNames.FILTERED_AND_NUMERIC_NAN.value.format(i, Consts.FileSubNames.Y_VAL.value),
                    Consts.FileNames.FILTERED_AND_NUMERIC_NAN.value.format(i, Consts.FileSubNames.Y_TEST.value)
                )

                secondStepPrep_dict[i].loadData(Consts.listAdditionalDataPreparation)
            secondStepPrep_dict[i].trainLabels = secondStepPrep_dict[i].trainLabels[["Vote"]]
            secondStepPrep_dict[i].valLabels = secondStepPrep_dict[i].valLabels[["Vote"]]
            secondStepPrep_dict[i].testLabels = secondStepPrep_dict[i].testLabels[["Vote"]]

            if Stages.do_relief:
                relief_dir = 'datasets\\{}\\'.format(i)
                N = 100
                tau = 0

                relief_chosen_set = relief.relief_alg(secondStepPrep_dict[i].trainData,
                                                      secondStepPrep_dict[i].trainLabels, N, tau)
                pd.DataFrame([f for f in relief_chosen_set]).to_csv(
                    relief_dir + "N_{}_tau_{}.csv".format(N, tau))
                secondStepPrep_dict[i].trainData = secondStepPrep_dict[i].trainData[relief_chosen_set]
                secondStepPrep_dict[i].valData = secondStepPrep_dict[i].valData[relief_chosen_set]
                secondStepPrep_dict[i].testData = secondStepPrep_dict[i].testData[relief_chosen_set]
            # Remove features with a very high correlation
            if Stages.do_removeAbove95Corr:
                secondStepPrep_dict[i].removeAbove95Corr()

            trainData = secondStepPrep_dict[i].trainData.copy()
            valData = secondStepPrep_dict[i].valData.copy()
            testData = secondStepPrep_dict[i].testData.copy()

            if Stages.do_sfs:
                # create a random forest for the sfs
                rClf = RandomForestClassifier()
                max_amount_of_features = 23
                bestFeatures = sfsAux(rClf, secondStepPrep_dict[i].trainData, secondStepPrep_dict[i].trainLabels,
                                      max_amount_of_features)
                print("Sfs chose: {}".format(",".join(bestFeatures)))
                secondStepPrep_dict[i].trainData = secondStepPrep_dict[i].trainData[bestFeatures]
                secondStepPrep_dict[i].valData = secondStepPrep_dict[i].valData[bestFeatures]
                secondStepPrep_dict[i].testData = secondStepPrep_dict[i].testData[bestFeatures]
                pd.DataFrame(bestFeatures).to_csv(
                    Consts.FileNames.SUMMARY.value.format(i, "Best_chosen_features_random_forest")
                )
                secondStepPrep_dict[i].trainData.to_csv(
                    Consts.FileNames.SUMMARY.value.format(i, "Best_chosen_train_random_forest")
                )
                secondStepPrep_dict[i].valData.to_csv(
                    Consts.FileNames.SUMMARY.value.format(i, "Best_val_chosen_random_forest")
                )
                secondStepPrep_dict[i].testData.to_csv(
                    Consts.FileNames.SUMMARY.value.format(i, "Best_test_chosen_random_forest")
                )
                # create svm for the sfs

                svm = SVC()
                bestFeatures = sfsAux(svm, trainData, secondStepPrep_dict[i].trainLabels,
                                      max_amount_of_features)
                trainData = trainData[bestFeatures]
                valData = valData[bestFeatures]
                testData = testData[bestFeatures]
                pd.DataFrame(bestFeatures).to_csv(
                    Consts.FileNames.SUMMARY.value.format(i, "Best_chosen_features_svm")
                )
                trainData.to_csv(
                    Consts.FileNames.SUMMARY.value.format(i, "Best_chosen_train_svm")
                )
                valData.to_csv(
                    Consts.FileNames.SUMMARY.value.format(i, "Best_val_chosen_svm")
                )
                testData.to_csv(
                    Consts.FileNames.SUMMARY.value.format(i, "Best_test_chosen_svm")
                )


if __name__ == "__main__":
    print("Executing the main frame")
    main()
