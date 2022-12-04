#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:22:23 2022

@author: jonathanchien
"""

import pandas as pd
data = pd.read_csv("/Users/jonathanchien/Documents/ECO481H Datahub/gss_clean.csv")
data.columns = data.columns.str.replace('regilion_importance', 'religion_importance')
new = data[["caseid", "age", "total_children", "sex", "marital_status", "education", "hh_size", "hh_type", "partner_education", 
                "average_hours_worked", "partner_main_activity", "income_family", "occupation", "children_in_household",  
                "religion_participation", "living_arrangement"]]

new_data = new.dropna()
descriptive_data = new_data.describe()
descriptive_data

print(new_data["hh_type"].unique())
df=new_data.groupby(by="hh_type")
df
#do grouping
#first I need to drop the "Other" and "Don't know" values in the data table
new_data.drop(new_data.loc[new_data['hh_type']=='Other'].index, inplace=True)
new_data.drop(new_data.loc[new_data['hh_type']=="Don't know"].index, inplace=True)
#after dropping the two values, then we rename the low-rise and high-rise apartment to apartment to generalize these 
#two values into one category
new_data['hh_type'] = new_data['hh_type'].replace(['High-rise apartment (5 or more stories)', 'Low-rise apartment (less than 5 stories)'], ['Apartment', 'Apartment'])
new_data['hh_type'] = new_data['hh_type'].replace(['Single detached house'], ['House'])



#transforming the family income categories into numbers in order to do descriptive statistics
new_data["income_family"].unique()
level_income = ["Less than $25,000", "$25,000 to $49,999", "$50,000 to $74,999", 
         "$75,000 to $99,999", "$100,000 to $ 124,999", "$125,000 and more"]
le_2 = LabelEncoder()
le_2.fit(level_income)
new_data["income_family"] = le_2.transform(new_data["income_family"])

#transforming the education categories into numbers in order to do descriptive statistics
new_data["education"].unique()
level_education = ["Bachelor's degree (e.g. B.A., B.Sc., LL.B.)",
       'High school diploma or a high school equivalency certificate',
       'College, CEGEP or other non-university certificate or di...',
       'University certificate, diploma or degree above the bach...',
       'Trade certificate or diploma',
       'Less than high school diploma or its equivalent',
       "University certificate or diploma below the bachelor's level"]

le_3 = LabelEncoder()
le_3.fit(level_education)
new_data["education"] = le_3.transform(new_data["education"])

#transforming the living arrangement categories into numbers in order to do descriptive statistics
new_data["living_arrangement"].unique()
level_living = ['Spouse and single child under 25 years of age',
       'Other living arrangement', 'Spouse only', 'Alone',
       'No spouse and single child under 25 years of age',
       'No spouse and single child 25 years of age or older',
       'Spouse and other',
       'Spouse and single child 25 years of age or older',
       'Living with one parent', 'Spouse and non-single child(ren)']

le_4 = LabelEncoder()
le_4.fit(level_living)
new_data["living_arrangement"] = le_4.transform(new_data["living_arrangement"])



grouped = new_data.groupby(by="living_arrangement").describe()
grouped

grouped1_forassign = new_data[["education", "income_family", "hh_size", "living_arrangement", "hh_type"]]
stats = grouped1_forassign.groupby(by="hh_type").describe()


print(new_data['living_arrangement'].value_counts())

ab = new_data[["education", "hh_type"]].groupby("education").describe()

ac = new_data[["living_arrangement", "hh_type"]].groupby("living_arrangement").describe()


#Actual Analysis 
from sklearn import tree 
from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle 
from sklearn.preprocessing import LabelEncoder

## Firstly, I am going to encode the categorial objects to integers
letters = ["Male", "Female", "Female", "Male", "Male"]
le_1 = LabelEncoder()
le_1.fit(letters)
#Checking what it has uniquely identified:
le_1.classes_
new_data["sex"] = le_1.transform(new_data["sex"])


#sort out to a new dataset 
only_hh = new_data['hh_type']
#drop the column caseid and hh_type
new_data = new_data.drop(labels='caseid', axis = 1)
new_data = new_data.drop(labels='hh_type', axis = 1)
#Then let us try to split the outcome variable to training and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(new_data, only_hh, test_size = 0.2)

#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler().fit(x_train)

#scaled_x_train = scaler.transform(x_train)
#scaled_x_test = scaler.transform(x_test)

#Save a list of the features we want to use in the classifier 
Features = list(new_data.columns)

df_shuffled = shuffle(new_data,random_state = 2)

##Decision tree 
from sklearn import tree 

dt_ = tree.DecisionTreeClassifier(random_state = 14)
dt_ = dt_.fit(x_train, y_train)

##Random Forest 
from sklearn.ensemble import RandomForestClassifier 

rf = RandomForestClassifier(max_depth = 2, random_state=0)
rf.fit(x_train, y_train)



