listSymbolicColumns = ['Most_Important_Issue', 'Main_transportation', 'Occupation']

listNonNumeric = ['Most_Important_Issue', 'Main_transportation', 'Occupation',
               'Looking_at_poles_results', 'Married', 'Gender', 'Voting_Time', 'Financial_agenda_matters',
               'Will_vote_only_large_party', 'Age_group']

setNumericFeatures = {'Avg_monthly_expense_when_under_age_21', 'AVG_lottary_expanses', 'Avg_Satisfaction_with_previous_vote',
                         'Garden_sqr_meter_per_person_in_residancy_area', 'Financial_balance_score_(0-1)', '%Of_Household_Income',
                         'Avg_government_satisfaction', 'Avg_education_importance', 'Avg_environmental_importance',
                         'Avg_Residancy_Altitude', 'Yearly_ExpensesK', '%Time_invested_in_work', 'Yearly_IncomeK',
                         'Avg_monthly_expense_on_pets_or_plants', 'Avg_monthly_household_cost', 'Phone_minutes_10_years',
                         'Avg_size_per_room', 'Weighted_education_rank', '%_satisfaction_financial_policy', 'Avg_monthly_income_all_years',
                         'Last_school_grades', 'Number_of_differnt_parties_voted_for', 'Political_interest_Total_Score',
                         'Number_of_valued_Kneset_members', 'Overall_happiness_score', 'Num_of_kids_born_last_10_years',
                         'Age_group_int', 'Occupation_Satisfaction', 'Will_vote_only_large_party_int'}

setGaussianFeatures ={ 'Avg_Satisfaction_with_previous_vote',
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

setUniformFeatures = {'Occupation_Satisfaction',
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

setSelectedFeatures = {'Number_of_valued_Kneset_members',
                       'Yearly_IncomeK',
                       'Overall_happiness_score',
                       'Avg_Satisfaction_with_previous_vote',
                       'Most_Important_Issue',
                       'Will_vote_only_large_party',
                       'Garden_sqr_meter_per_person_in_residancy_area',
                       'Weighted_education_rank'}

listFixNegateVals = [
    'Avg_monthly_expense_when_under_age_21',
    'AVG_lottary_expanses',
    'Avg_Satisfaction_with_previous_vote'
]

listAdditionalDataPreparation = ["validation", "test", ""]


inf = 10000
RAW_FILE_PATH = 'ElectionsData.csv'
RAW_SPLIT_FILE_DIRECTORY = "datasets"
RAW_SPLIT_FILE_PATH = "datasets/{}/{}{}"
