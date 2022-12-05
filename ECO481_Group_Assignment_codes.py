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

#Need to separate the education into another category 
#Split the education level to two different education categories
#Higher education is education levels that are post-secondary
new_data['education'].unique()
new_data['education'] = new_data['education'].replace(['Less than high school diploma or its equivalent', 
                                                       'High school diploma or a high school equivalency certificate' 
                                                       ], ['Lower Education', 'Lower Education'])
new_data['education'] = new_data['education'].replace(['Trade certificate or diploma', 
                                                       "Bachelor's degree (e.g. B.A., B.Sc., LL.B.)",
                                                       "University certificate or diploma below the bachelor's level", 
                                                       'University certificate, diploma or degree above the bach...',
                                                       'College, CEGEP or other non-university certificate or di...'
                                                       ], ['Higher Education', 'Higher Education', 'Higher Education',
                                                           'Higher Education', 'Higher Education'])
#Categorize the partner's education into higher and lower educations as well
new_data["partner_education"].unique()
new_data["partner_education"] = new_data["partner_education"].replace(['Less than high school diploma or its equivalent', 
                                                       'High school diploma or a high school equivalency certi...' 
                                                       ], ['Lower Education', 'Lower Education'])
new_data["partner_education"] = new_data["partner_education"].replace(['Trade certificate or diploma', 
                                                       "Bachelor's degree (e.g. B.A., B.Sc., LL.B.)",
                                                       "University certificate or diploma below the bachelor's level", 
                                                       'University certificate, diploma or degree above the ba...',
                                                       'College, CEGEP or other non-university certificate or d...'
                                                       ], ['Higher Education', 'Higher Education', 'Higher Education',
                                                           'Higher Education', 'Higher Education'])
#Categorize martial status
new_data["marital_status"].unique()
new_data["marital_status"] = new_data["marital_status"].replace(['Married'], ['Living Together'])
new_data["marital_status"] = new_data["marital_status"].replace(['Single, never married', 'Living common-law',
                                                               'Separated', 'Divorced', 'Widowed'], 
                                                              ['Living Alone', 'Living Alone', 'Living Alone',
                                                               'Living Alone', 'Living Alone'])

#Categorize average working hours
new_data["partner_main_activity"].unique()
#First drop the values who has "Dont know " average working hour values
new_data.drop(new_data.loc[new_data["partner_main_activity"]=="Other"].index, inplace=True)
new_data["partner_main_activity"] = new_data["partner_main_activity"].replace(['Going to school', 'Retired', 'Long term illness', 
                                                                               'Household work',
                                                                               'Maternity/paternity/parental leave', 'Caring for children',
                                                                               'Volunteering or care-giving other than for children'], 
                                                                            ['Unemployed', 'Unemployed', 'Unemployed', 'Unemployed', 'Unemployed',
                                                                             'Unemployed', 'Unemployed'])
new_data["partner_main_activity"] = new_data["partner_main_activity"].replace(['Working at a paid job or business', 'Looking for paid work'], 
                                                                            ['Employed', 'Employed'])

#Categorize income
new_data["income_family"].unique()
new_data["income_family"] = new_data["income_family"].replace(['$125,000 and more', '$100,000 to $ 124,999'], 
                                                              ['High Income', 'High Income'])
new_data["income_family"] = new_data["income_family"].replace(['$25,000 to $49,999', '$75,000 to $99,999', 
                                                               '$50,000 to $74,999', 'Less than $25,000'], 
                                                              ['Low Income', 'Low Income', 'Low Income',
                                                               'Low Income'])

#Categorize occupation
new_data["occupation"].unique()
#First drop all the values with "Uncodable"
new_data.drop(new_data.loc[new_data["occupation"]=='Uncodable'].index, inplace=True) 
new_data["occupation"] = new_data["occupation"].replace(['Trades, transport and equipment operators and related oc...', 
                                                         'Occupations in manufacturing and utilities',
                                                         'Natural resources, agriculture and related production oc...'], 
                                                        ['Industrial Industry', 'Industrial Industry', 'Industrial Industry'])
new_data["occupation"] = new_data["occupation"].replace(['Sales and service occupations',
                                                         'Business, finance, and administration occupations',
                                                         'Occupations in education, law and social, community and ...',
                                                         'Management occupations',
                                                         'Occupations in art, culture, recreation and sport',
                                                         'Natural and applied sciences and related occupations',
                                                         'Health occupations'], 
                                                        ['Service Industry', 'Service Industry', 'Service Industry',
                                                         'Service Industry', 'Service Industry', 'Service Industry',
                                                         'Service Industry'])

#Categorize have children or not
new_data["children_in_household"].unique()
new_data["children_in_household"] = new_data["children_in_household"].replace(['No child'], ['No children'])
new_data["children_in_household"] = new_data["children_in_household"].replace(['Two children', 'One child', 
                                                                               'Three or more children'], 
                                                                              ['Have children', 'Have children', 'Have children'])

#Categorize religious participation
new_data["religion_participation"].unique()
#First drop all the values with "Don't know"
new_data.drop(new_data.loc[new_data["religion_participation"]== "Don't know"].index, inplace=True) 
new_data["religion_participation"] = new_data["religion_participation"].replace(['Not at all'], ['No religious participation'])
new_data["religion_participation"] = new_data["religion_participation"].replace(['Once or twice a year', 'At least once a month',
                                                                                 'At least 3 times a year', 'At least once a week'], 
                                                                                ['Have religious participation', 
                                                                                 'Have religious participation', 
                                                                                 'Have religious participation', 
                                                                                 'Have religious participation'])

#Categorize religious participation
new_data["living_arrangement"].unique()
#First drop all the values with "Don't know"
new_data.drop(new_data.loc[new_data["living_arrangement"]== 'Other living arrangement'].index, inplace=True) 
new_data["living_arrangement"] = new_data["living_arrangement"].replace(['Not at all'], ['No religious participation'])
new_data["living_arrangement"] = new_data["living_arrangement"].replace(['Once or twice a year', 'At least once a month',
                                                                                 'At least 3 times a year', 'At least once a week'], 
                                                                                ['Have religious participation', 
                                                                                 'Have religious participation', 
                                                                                 'Have religious participation', 
                                                                                 'Have religious participation'])

#Categorize 
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



