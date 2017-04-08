#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from copy import deepcopy
import numpy as np
import pandas as pd
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'long_term_incentive',
                 'deferred_income', 'other', 'expenses', 'total_payments',
                 'restricted_stock', 'restricted_stock_deferred',
                 'total_stock_value', 'from_poi_to_this_person',
                 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data = pickle.load(data_file)

### Task 2: Remove outliers

cleaned_data = deepcopy(data)

for person in cleaned_data:
    for key in cleaned_data[person]:
        for row in cleaned_data:
            if (cleaned_data[row][key] == "NaN"):
                cleaned_data[row][key] =0
    break

del cleaned_data['BELFER ROBERT']
del cleaned_data['BHATNAGAR SANJAY']
    
financial_feature = ['salary', 'bonus', 'long_term_incentive', 'deferred_income', 
                     'deferral_payments', 'loan_advances', 'other', 'expenses', 
                     'director_fees', 'total_payments',
                     'exercised_stock_options', 'restricted_stock', 
                     'restricted_stock_deferred', 'total_stock_value']
email_feature = ['to_messages', 'from_messages', 
                 'from_poi_to_this_person', 'from_this_person_to_poi', 
                 'shared_receipt_with_poi']
feature_list = ['poi'] +  financial_feature + ['email_address'] + email_feature
feature_list_no_email_no_poi = financial_feature + email_feature

data_dictionary = {}
for key in feature_list:
    feature_data = []
    for person in cleaned_data:
        feature_data.append(cleaned_data[person][key])
    data_dictionary[key] = feature_data

name_list = []
for person in cleaned_data:
    name_list.append(person)    
data_dictionary['name'] = name_list

df = pd.DataFrame(data_dictionary)
df = df[(['name'] + feature_list)]

for key in financial_feature:
    df[key] = (np.sign(df[key]) 
             * np.log10(abs(df[key]) + 1))
    
for key in feature_list_no_email_no_poi:
    df[key] = (df[key] - np.mean(df[key])) / np.std(df[key])
    
final_df = deepcopy(df)
temp_data = final_df[feature_list].as_matrix()
names = final_df['name'].as_matrix()

my_dataset = {}
for i, name in enumerate(names):
    my_dataset[name] = {}
    for j, feature in enumerate(feature_list):
        my_dataset[name][feature] = temp_data[i][j]



### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.



### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=1)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
