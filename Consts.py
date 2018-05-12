from enum import Enum

inf = 10000
maxLeafNodes = 20
INDEX_COL = 'index_col'
VOTE_STR = 'Vote'
VOTE_INT = 'Vote_int'

setSelectedFeatures = {
    'Number_of_valued_Kneset_members',
    'Yearly_IncomeK',
    'Overall_happiness_score',
    'Avg_Satisfaction_with_previous_vote',
    'Most_Important_Issue',
    'Will_vote_only_large_party',
    'Garden_sqr_meter_per_person_in_residancy_area',
    'Weighted_education_rank',
    INDEX_COL
}

_listSymbolicColumns = [
    'Most_Important_Issue',
    'Main_transportation',
    'Occupation'
]

listSymbolicColumns = [feature for feature in _listSymbolicColumns if feature in setSelectedFeatures]

_listNonNumeric = [
    'Most_Important_Issue',
    'Main_transportation',
    'Occupation',
    'Looking_at_poles_results',
    'Married',
    'Gender',
    'Voting_Time',
    'Financial_agenda_matters',
    'Will_vote_only_large_party',
    'Age_group'
]

listNonNumeric = [feature for feature in _listNonNumeric if feature in setSelectedFeatures]


_setNumericFeatures = {
    'Avg_monthly_expense_when_under_age_21',
    'AVG_lottary_expanses',
    'Avg_Satisfaction_with_previous_vote',
    'Garden_sqr_meter_per_person_in_residancy_area',
    'Financial_balance_score_(0-1)',
    '%Of_Household_Income',
    'Avg_government_satisfaction',
    'Avg_education_importance',
    'Avg_environmental_importance',
    'Avg_Residancy_Altitude',
    'Yearly_ExpensesK',
    '%Time_invested_in_work',
    'Yearly_IncomeK',
    'Avg_monthly_expense_on_pets_or_plants',
    'Avg_monthly_household_cost',
    'Phone_minutes_10_years',
    'Avg_size_per_room',
    'Weighted_education_rank',
    '%_satisfaction_financial_policy',
    'Avg_monthly_income_all_years',
    'Last_school_grades',
    'Number_of_differnt_parties_voted_for',
    'Political_interest_Total_Score',
    'Number_of_valued_Kneset_members',
    'Overall_happiness_score',
    'Num_of_kids_born_last_10_years',
    'Age_group_int',
    'Occupation_Satisfaction',
    'Will_vote_only_large_party_int'
}

setNumericFeatures = { feature for feature in _setNumericFeatures if feature in setSelectedFeatures}

_setGaussianFeatures ={
    'Avg_Satisfaction_with_previous_vote',
     'Garden_sqr_meter_per_person_in_residancy_area',
     'Yearly_IncomeK',
     'Avg_monthly_expense_on_pets_or_plants',
     'Avg_monthly_household_cost',
     'Phone_minutes_10_years',
     'Avg_size_per_room',
     'Weighted_education_rank',
     'Number_of_differnt_parties_voted_for',
     'Political_interest_Total_Score',
     'Overall_happiness_score',
     'Num_of_kids_born_last_10_years',
     'Number_of_valued_Kneset_members',
     'Avg_monthly_income_all_years',
     'AVG_lottary_expanses',
     'Avg_monthly_expense_when_under_age_21',
}
setGaussianFeatures = {feature for feature in _setGaussianFeatures if feature in setSelectedFeatures}

_setUniformFeatures = {
    'Occupation_Satisfaction',
    'Avg_government_satisfaction',
    'Avg_education_importance',
    'Avg_environmental_importance',
    'Avg_Residancy_Altitude',
    'Yearly_ExpensesK',
    '%Time_invested_in_work',
    "Last_school_grades",
    '%_satisfaction_financial_policy',
    '%Of_Household_Income'
}

setUniformFeatures = {feature for feature in _setUniformFeatures if feature in setSelectedFeatures}

_listFixNegateVals = [
    'Avg_monthly_expense_when_under_age_21',
    'AVG_lottary_expanses',
    'Avg_Satisfaction_with_previous_vote'
]

listFixNegateVals = [feature for feature in _listFixNegateVals if feature in setSelectedFeatures]

listAdditionalDataPreparation = ["validation", "test", ""]

listRandomStates = [376674226, 493026216, 404629562, 881225405]

MAP_VOTE_TO_NUMERIC = {
    'Greens': 10,
    'Pinks': 9,
    'Purples': 8,
    'Blues': 7,
    'Whites': 6,
    'Browns': 5,
    'Yellows': 4,
    'Reds': 3,
    'Turquoises': 2,
    'Greys': 1,
    'Oranges': 11
}

class DataTypes(Enum):
    TEST = 'test'
    VAL = 'val'
    TRAIN = 'train'

class ClassifierTypes(Enum):
    TREE = "tree"
    SVM = "svm"
    RANDOM_FOREST = "random_forest"

class FileSubNames(Enum):
    X_TRAIN = 'X_train'
    X_VAL = 'X_val'
    X_TEST = 'X_test'
    Y_TRAIN = 'Y_train'
    Y_VAL = 'Y_val'
    Y_TEST = 'Y_test'

class DirNames(Enum):
    DATA_SETS = 'datasets'
    DATA_SETS_I = 'datasets/{}'
    RAW_AND_SPLITED = "datasets/{}/raw_spited"
    RAW_AND_FILTERED = "datasets/{}/raw_and_filtered"
    FILTERED_AND_NUMERIC_NAN = "datasets/{}/filtered_and_numeric_nan"
    FILTERED_AND_NUMERIC_NONAN = "datasets/{}/filtered_and_numeric_nonan"
    FILTERED_AND_SCALED = "datasets/{}/filtered_and_scaled"
    SUMMARY = "datasets/{}/summary"

per_file = "/{}.csv"

class FileNames(Enum):
    FROM_INTERNET = ""
    RAW_FILE_PATH = 'ElectionsData.csv'
    RAW_AND_SPLITED = DirNames.RAW_AND_SPLITED.value + per_file
    RAW_AND_FILTERED  = DirNames.RAW_AND_FILTERED.value + per_file
    FILTERED_AND_NUMERIC_NAN = DirNames.FILTERED_AND_NUMERIC_NAN.value + per_file
    FILTERED_AND_NUMERIC_NONAN = DirNames.FILTERED_AND_NUMERIC_NONAN.value + per_file
    FILTERED_AND_SCALED = DirNames.FILTERED_AND_SCALED.value + per_file
    SUMMARY = DirNames.SUMMARY.value + per_file

