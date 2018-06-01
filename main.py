import os

import Consts
from ElectionsDataPreperation import ElectionsDataPreperation as EDP, DataSplit
from modeling import ex_3
from scale_data import ScaleData


class Stages:
    # Stages:
    do_print = True
    do_get_raw_data = False
    do_filter_features = False
    do_swap_to_numeric = False
    do_fix_nan_and_outliers = False
    do_scale = False
    do_feature_selection = False
    do_feature_selection_load_data = False
    do_removeAbove95Corr = False
    do_sfs = False
    do_relief = False
    get_correlations = False
    # EX3
    use_the_same_model_for_all_tasks = True
    use_multi_models_for_tasks: bool = True
    show_learning_curves: bool = True
    view_decision_tree: bool = True
    print_ex3: bool = True


amount_of_sets = 1

def load_edp_from(base: str, set: int) -> EDP:
    edp = EDP(
        base.format(set, Consts.FileSubNames.X_TRAIN.value),
        base.format(set, Consts.FileSubNames.X_VAL.value),
        base.format(set, Consts.FileSubNames.X_TEST.value),
        base.format(set, Consts.FileSubNames.Y_TRAIN.value),
        base.format(set, Consts.FileSubNames.Y_VAL.value),
        base.format(set, Consts.FileSubNames.Y_TEST.value)
    )
    edp.loadData()

    return edp

def create_files():
    for d in Consts.DirNames:
        if d == Consts.DirNames.DATA_SETS:
            if not os.path.isdir(d.value):
                os.mkdir(d.value)

        else:
            for i in range(1, 3):
                if not os.path.isdir(d.value.format(i)):
                    os.mkdir(d.value.format(i))


def log(msg):
    if Stages.do_print:
        print(msg)

def ex2_remainder():
    create_files()

    # FIRST STEP: Get the data and split it in to 2 groups of 3 data sets.
    # we need to bring the initial file only once. while working on it, it is rather efficient to work on local files
    # yet we'd like to be able to get the files and fall threw these steps again if needed.
    if Stages.do_get_raw_data:
        log("Stage 1: Importing the data")
        ds = DataSplit(Consts.FileNames.RAW_FILE_PATH.value)
        ds.saveDataSetsToCsv()

    # SECOND STEP: Prepare the data for work.
    secondStepPrep_dict = dict()
    scaleData_dict = dict()

    if Stages.do_filter_features:
        log("Stage 2: Filtering Labels")

        for i in range(1, amount_of_sets + 1):
            # start the preparing data class
            secondStepPrep_dict[i] = load_edp_from(Consts.FileNames.RAW_AND_SPLITED.value, i)
            secondStepPrep_dict[i].filterFeatures(list(Consts.DataTypes))

            secondStepPrep_dict[i].save_data(Consts.FileNames.RAW_AND_FILTERED.value, i)
            secondStepPrep_dict[i].save_labels(Consts.FileNames.RAW_AND_FILTERED.value, i)

    if Stages.do_swap_to_numeric:
        log("Stage 3: Swapping strings to numeric values")

        for i in range(1, amount_of_sets + 1):
            # start the preparing data class
            secondStepPrep_dict[i] = load_edp_from(Consts.FileNames.RAW_AND_FILTERED.value, i)
            secondStepPrep_dict[i]._changeStringToValues(list(Consts.DataTypes))

    if Stages.do_fix_nan_and_outliers:

        log("Stage 4: Fixing nan and outliers")

        for i in range(1, amount_of_sets + 1):
            secondStepPrep_dict[i] = load_edp_from(Consts.FileNames.FILTERED_AND_NUMERIC_NAN.value, i)

            secondStepPrep_dict[i].fix_nan_and_outliers()
            secondStepPrep_dict[i].save_labels(Consts.FileNames.FILTERED_AND_NUMERIC_NONAN.value, i)

    if Stages.do_scale:
        log("Stage 5: Scale the data")
        for i in range(1, amount_of_sets + 1):
            # start the preparing data class
            secondStepPrep_dict[i] = load_edp_from(Consts.FileNames.FILTERED_AND_NUMERIC_NONAN.value, i)

            initial_corr = secondStepPrep_dict[i].trainData.corr()
            if Stages.get_correlations:
                initial_corr.to_csv(Consts.FileNames.SUMMARY.value.format(i, 'initial_corr'))

            # scale the data
            scaleData_dict[i] = ScaleData()  # type: ScaleData
            scaleData_dict[i].scale_train(secondStepPrep_dict[i].trainData)
            scaleData_dict[i].scale_test(secondStepPrep_dict[i].valData)
            scaleData_dict[i].scale_test(secondStepPrep_dict[i].testData)
            # scaleData_dict[i].scale_test(secondStepPrep_dict[i].testData)
            secondStepPrep_dict[i].save_data(Consts.FileNames.FILTERED_AND_SCALED.value, i)
            secondStepPrep_dict[i].save_labels(Consts.FileNames.FILTERED_AND_SCALED.value, i)

            second_corr = secondStepPrep_dict[i].trainData.corr()
            if Stages.get_correlations:
                second_corr.to_csv(Consts.FileNames.SUMMARY.value.format(i, 'Scaled_corr_diff'))
                (second_corr - initial_corr).abs().to_csv(Consts.FileNames.SUMMARY.value.format(i, 'Scaled_corr_diff'))

def main():
   ex2_remainder()
   ex_3(
       use_the_same_model_for_all_tasks=Stages.use_the_same_model_for_all_tasks,
       show_learning_curves=Stages.show_learning_curves,
       use_multi_models_for_tasks=Stages.use_multi_models_for_tasks,
       view_decision_tree=Stages.view_decision_tree,
       print_ex3=Stages.print_ex3
   )


if __name__ == "__main__":
    print("Executing the main frame")
    main()
