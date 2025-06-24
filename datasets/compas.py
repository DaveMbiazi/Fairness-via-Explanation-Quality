# path to the dataset
path = "...."

# columns
columns = ['sex','age','race','juv_fel_count','juv_misd_count','juv_other_count','priors_count','c_charge_degree','low_risk']

# all training columns
train_cols = ['age','race','sex','juv_fel_count','priors_count','c_charge_degree']

# label column
label = 'low_risk'

# sensitive columns
sensitive_attributes = ["race"]

use_sensitive = 1

# whether data already contains splits
already_split = False

# list of all categorical columns
categorical_columns = ['race', 'sex', 'c_charge_degree']

# balanced groups
balance_groups = 0

has_header = True