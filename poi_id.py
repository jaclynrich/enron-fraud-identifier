import sys
import pickle
sys.path.append("tools/")

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from tester import dump_classifier_and_data

import pandas as pd
pd.options.mode.use_inf_as_na = True
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

### Load and explore the data
# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

# Load the data into a dataframe for exploration
df = pd.DataFrame.from_dict(data_dict).T

# Replace 'NaN' in email_address field with empty string
df['email_address'].replace(to_replace='NaN', value = '', inplace=True)

# Replace all other 'NaN' with NaN
df.replace(to_replace='NaN', value = np.nan, inplace=True)

# Convert columns to numeric
df = df.apply(lambda x: pd.to_numeric(x, errors='ignore'))

target = 'poi'

### Removing Records
# Remove periods from names to make them consistent
df.index = df.index.map(lambda x: x.replace('.', ''))

# Drop non-employee records
df.drop(['THE TRAVEL AGENCY IN THE PARK', 'TOTAL'], inplace = True)

# Remove email since it is basically equivalent to an ID and cannot be used
# in the model
df.drop('email_address', axis = 1, inplace = True)

# Drop record with all nulls
df.drop('LOCKHART EUGENE E', inplace = True)

# Fill in missing values with 0
df = df.fillna(0)

### Feature Engineering
# These feature describes the proportion of communication that involves
# persons of interest
df['prop_poi_communication'] = (df['from_poi_to_this_person'] + \
                                df['from_this_person_to_poi']) / \
                               (df['to_messages'] + df['from_messages'])
df['shared_receipt_with_poi_prop'] = df['shared_receipt_with_poi'] / \
                                    (df['to_messages'] + df['from_messages'])

# These features describe the proportion of total_payments that each individual
# feature is
df['bonus_prop'] = df['bonus'] / df['total_payments']
df['other_prop'] = df['other'] / df['total_payments']
df['expenses_prop'] = df['expenses'] / df['total_payments']

# These features describe the proportion of total_stock_value that each
# individual feature is
df['exercised_stock_options_prop'] = df['exercised_stock_options'] / \
                                     df['total_stock_value']
df['restricted_stock_prop'] = df['restricted_stock'] / df['total_stock_value']

# Get new features list
all_features = [feat for feat in list(df.columns) if feat != 'poi']

# After possible division by zero fill inf and -inf with 0
df = df.fillna(0)

### Feature Selection
feature_selector = SelectKBest(f_classif, k='all').fit(df[all_features],
                                                       df[target])
feat_scores = list(zip(feature_selector.scores_,
                   df[all_features].columns[feature_selector.get_support()]))
feat_scores.sort(key=lambda tup: tup[0], reverse = True)
feats_sorted = [tup[1] for tup in feat_scores]

### Building the model
# Generic classification function that returns training and validation metrics
def classify(model, data, features, target):

    skf = StratifiedShuffleSplit(n_splits=10, random_state = 42)

    tr_metrics = {}
    val_metrics = {}

    metric_names = ['acc', 'prec', 'recall', 'f1']

    for name in metric_names:
            tr_metrics[name] = []
            val_metrics[name] = []

    for train, test in skf.split(data[features], data[target]):
        X_tr = (data[features].iloc[train,:])
        y_tr = data[target].iloc[train]
        X_val = data[features].iloc[test,:]
        y_val = data[target].iloc[test]

        # Train the model and record metrics
        model.fit(X_tr, y_tr)
        pred_tr = model.predict(X_tr)

        tr_metrics['acc'].append(accuracy_score(pred_tr, y_tr))
        tr_metrics['prec'].append(precision_score(pred_tr, y_tr))
        tr_metrics['recall'].append(recall_score(pred_tr, y_tr))
        tr_metrics['f1'].append(f1_score(pred_tr, y_tr))

        # Make predictions on the validation set and record metrics
        pred = model.predict(X_val)

        val_metrics['acc'].append(accuracy_score(pred, y_val))
        val_metrics['prec'].append(precision_score(pred, y_val))
        val_metrics['recall'].append(recall_score(pred, y_val))
        val_metrics['f1'].append(f1_score(pred, y_val))

    tr_mean_metrics = []
    val_mean_metrics = []

    # Only keep averages
    for name in metric_names:
        tr_mean_metrics.append(np.mean(tr_metrics[name]))
        val_mean_metrics.append(np.mean(val_metrics[name]))

    return tr_mean_metrics, val_mean_metrics

# Run the classifier for 1:num_feat number of features
def model_by_num_feat(model, data, features, target, num_feat):
    tr_metrics_feat = {}
    val_metrics_feat = {}
    for i in range(1, num_feat + 1):
        tr_metrics, val_metrics = classify(model, data, features[:i], target)
        tr_metrics_feat[i] = tr_metrics
        val_metrics_feat[i] = val_metrics

    tr_metrics_df = pd.DataFrame(tr_metrics_feat).T
    tr_metrics_df.columns = ['tr_accuracy', 'tr_precision', 'tr_recall', 'tr_f1']

    val_metrics_df = pd.DataFrame(val_metrics_feat).T
    val_metrics_df.columns = ['cv_accuracy', 'cv_precision', 'cv_recall', 'cv_f1']

    metrics_df = pd.concat([tr_metrics_df, val_metrics_df], axis = 1)
    return metrics_df[['tr_accuracy', 'cv_accuracy', 'tr_precision', 'cv_precision',
                       'tr_recall', 'cv_recall', 'tr_f1', 'cv_f1']]

### Final classifier

# Make data into a dictionary again to pass to tester
df_dict = df.to_dict('index')

# Since it takes a long time to run, I did not include the GridSearchCV
# The results from it are passed in directly
"""
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 200, num = 11)]
criterion = ['gini', 'entropy']
max_features = ['auto', 'log2', None]
bootstrap = [True, False]

param_grid = {'n_estimators': n_estimators,
              'criterion': criterion,
              'max_features': max_features,
              'bootstrap': bootstrap,
              'n_jobs': [-1]}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, scoring='recall_macro')
grid_search_clf = grid_search.fit(df[feats_sorted[:3]], df[target])

 """
best_params = {'bootstrap': False,
               'criterion': 'gini',
               'max_features': 'log2',
               'n_estimators': 120,
               'n_jobs': -1}

clf_rf = RandomForestClassifier(**best_params)
chosen_feats = ['poi']
chosen_feats.extend(feats_sorted[:3])

if __name__ == '__main__':
    # Dump final chosen classifier
    dump_classifier_and_data(clf_rf, data_dict, chosen_feats)
    print('Dumped final classifier and data')
