import folktables
import numpy as np

columns = [
    "AGEP", "COW", "SCHL", "MAR", "OCCP",
    "POBP", "RELP", "WKHP", "SEX", "RAC1P", 
    "ESR"  # Employment status recode
]

# all training columns - Exclude the target column
train_cols = [
    "AGEP", "COW", "SCHL", "MAR", "OCCP",
    "POBP", "RELP", "WKHP", "SEX", "RAC1P"
]

# label column - Specific to the task, e.g., ESR for employment status
label = 'ESR'

# sensitive columns 
sensitive_attributes = ["SEX"]

use_sensitive = False 

already_split = False

# list of all categorical columns
categorical_columns = {
     """ "COW": {
        1.0: (
            "Employee of a private for-profit company or"
            "business, or of an individual, for wages,"
            "salary, or commissions"
        ),
        2.0: (
            "Employee of a private not-for-profit, tax-exempt,"
            "or charitable organization"
        ),
        3.0: "Local government employee (city, county, etc.)",
        4.0: "State government employee",
        5.0: "Federal government employee",
        6.0: (
            "Self-employed in own not incorporated business,"
            "professional practice, or farm"
        ),
        7.0: (
            "Self-employed in own incorporated business,"
            "professional practice or farm"
        ),
        8.0: "Working without pay in family business or farm",
        9.0: "Unemployed and last worked 5 years ago or earlier or never worked",
    }, """
    "SCHL": {
        1.0: "No schooling completed",
        2.0: "Nursery school, preschool",
        3.0: "Kindergarten",
        4.0: "Grade 1",
        5.0: "Grade 2",
        6.0: "Grade 3",
        7.0: "Grade 4",
        8.0: "Grade 5",
        9.0: "Grade 6",
        10.0: "Grade 7",
        11.0: "Grade 8",
        12.0: "Grade 9",
        13.0: "Grade 10",
        14.0: "Grade 11",
        15.0: "12th grade - no diploma",
        16.0: "Regular high school diploma",
        17.0: "GED or alternative credential",
        18.0: "Some college, but less than 1 year",
        19.0: "1 or more years of college credit, no degree",
        20.0: "Associate's degree",
        21.0: "Bachelor's degree",
        22.0: "Master's degree",
        23.0: "Professional degree beyond a bachelor's degree",
        24.0: "Doctorate degree",
    },
    "MAR": {
        1.0: "Married",
        2.0: "Widowed",
        3.0: "Divorced",
        4.0: "Separated",
        5.0: "Never married or under 15 years old",
    },
    #"SEX": {1.0: "Male", 2.0: "Female"},
    "RAC1P": {
        1.0: "White alone",
        2.0: "Black or African American alone",
        3.0: "American Indian alone",  
        4.0: "Alaska Native alone",
        5.0: (
            "American Indian and Alaska Native tribes specified;"
            "or American Indian or Alaska Native,"
            "not specified and no other"
        ),
        6.0: "Asian alone",
        7.0: "Native Hawaiian and Other Pacific Islander alone",
        8.0: "Some Other Race alone",
        9.0: "Two or More Races",
    },
    "RELP": {
        0.0: "Reference person",
        1.0: "Husband/wife",
        2.0: "Nursery school, preschool",
        3.0: "Kindergarten",
        4.0: "Grade 1",
        5.0: "Grade 2",
        6.0: "Grade 3",
        7.0: "Grade 4",
        8.0: "Grade 5",
        9.0: "Grade 6",
        10.0: "Grade 7",
        11.0: "Grade 8",
        12.0: "Grade 9",
        13.0: "Grade 10",
        14.0: "Grade 11",
        15.0: "12th grade - no diploma",
        16.0: "Regular high school diploma",
        17.0: "GED or alternative credential",
    },
    "MIL": {
        1.0: "Now on active duty",
        2.0: "On active duty in the past, but not now",
        3.0: "Only active duty for training in reserves", 
        4.0: "Never served in the military"
    }
}

state = ["AL"]

# ACSEmployment problem definition
task = folktables.BasicProblem(
    features=[
        'AGEP',
        'SCHL',
        #'MAR',
        'RELP',
        'DIS',
        'MIL',
        #'SEX',
        #'RAC1P',
    ],
    target='ESR',
    target_transform=lambda x: x == 1,
    group='AGEP',
    group_transform=lambda x: (x <= 30).astype(int),
    preprocess=lambda x: x,
    postprocess=lambda x: np.nan_to_num(x, -1)
)